import os
import csv
import argparse

import gc
import torch
import numpy as np
from tqdm import tqdm

from data.data_loader_aug import SpectrogramDataset, AudioDataLoader
from data.utils import get_cer_wer
from decoder import GreedyDecoder
from model import DeepSpeech
from opts import add_decoder_args, add_inference_args

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--cache-dir', metavar='DIR',
                    help='path to save temp audio', default='data/cache/')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--errors', action="store_true", help="print error report")
parser.add_argument('--best', action="store_true", help="print best results")
parser.add_argument('--norm', default='max_frame', action="store",
                    help='Normalize sounds. Choices: "mean", "frame", "max_frame", "none"')
parser.add_argument('--data-parallel', dest='data_parallel', action='store_true',
                    help='Use data parallel')
parser.add_argument('--report-file', metavar='DIR', default='data/test_report.csv', help="Filename to save results")
no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
parser = add_decoder_args(parser)
args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    package = torch.load(args.continue_from,
                         map_location=lambda storage, loc: storage)
    # model = DeepSpeech.load_model(args.model_path)
    model = DeepSpeech.load_model_package(package)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.data_parallel:
        model = torch.nn.DataParallel(model).to(device)
        print('Using DP')
    model.eval()

    print(model)

    # zero-out the aug bits
    audio_conf = {**audio_conf,
                  'noise_prob': 0,
                  'aug_prob_8khz': 0,
                  'aug_prob_spect': 0,
                  'phoneme_count': 0,
                  'phoneme_map': None}

    print(audio_conf)

    report_file = None
    if args.report_file:
        os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
        report_file = csv.writer(open(args.report_file, 'wt'))
        report_file.writerow(['wav', 'text', 'transcript', 'offsets', 'CER', 'WER'])

    if args.decoder == "beam":
        from .decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      manifest_filepath=args.test_manifest,
                                      cache_path=args.cache_dir,
                                      labels=labels,
                                      normalize=args.norm,
                                      augment=False)

    # import random;random.shuffle(test_dataset.ids)

    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    # check calculation
    avg_total_wer, avg_total_cer = 0, 0

    processed_files = []
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, filenames, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        inputs = inputs.to(device)

        # print(inputs.shape, inputs.is_cuda, input_sizes.shape, input_sizes.is_cuda)
        model_outputs = model(inputs, input_sizes)
        # ignore phoneme outputs
        if len(model_outputs) == 5:
            out0, out, output_sizes, _, _ = model_output
        else:
            out0, out, output_sizes = model_outputs
        del inputs, targets, input_percentages, target_sizes, model_outputs

        if decoder is None: continue
        decoded_output, _ = decoder.decode(out.data, output_sizes.data)
        target_strings = target_decoder.convert_to_strings(split_targets)

        out_raw_cpu = out0.cpu().numpy()
        out_softmax_cpu = out.cpu().numpy()
        sizes_cpu = output_sizes.cpu().numpy()
        for x in tqdm(range(len(target_strings))):
            transcript, reference = decoded_output[x][0], target_strings[x][0]

            wer, cer, wer_ref, cer_ref = get_cer_wer(decoder, transcript[:2000], reference[:2000])

            if args.output_path:
                # add output to data array, and continue
                import pickle
                with open(filenames[x]+'.ts', 'wb') as f:
                    results = {
                        'logits': out_raw_cpu[x, :sizes_cpu[x]],
                        'probs': out_softmax_cpu[x, :sizes_cpu[x]],
                        'len': sizes_cpu[x],
                        'transcript': transcript,
                        'reference': reference,
                        'filename': filenames[x],
                        'wer': wer / wer_ref,
                        'cer': cer / cer_ref,
                    }
                    pickle.dump(results, f, protocol=4)
                    del results
                # continue
                processed_files.append(filenames[x] + '.ts')

            if args.verbose:
                print("Ref:", reference)
                print("Hyp:", transcript)
                print("Wav:", filenames[x])
                print("WER:", "{:.2f}".format(100 * wer / wer_ref), "CER:", "{:.2f}".format(100 * cer / cer_ref), "\n")
            elif args.errors:
                if cer / cer_ref > 0.5 and transcript.strip():
                    # print("FN:", )
                    print("Ref:", reference)
                    print("Hyp:", transcript)
                    print("Wav:", filenames[x])
                    print("WER:", "{:.2f}".format(100 * wer / wer_ref), "CER:", "{:.2f}".format(100 * cer / cer_ref),
                          "\n")
            elif args.best:
                if cer / cer_ref < 0.15:
                    # print("FN:", )
                    print("Ref:", reference)
                    print("Hyp:", transcript)
                    print("Wav:", filenames[x])
                    print("WER:", "{:.2f}".format(100 * wer / wer_ref), "CER:", "{:.2f}".format(100 * cer / cer_ref),
                          "\n")

            if report_file:
                # report_file.write_row(['wav', 'text', 'transcript', 'offsets', 'CER', 'WER'])

                report_file.writerow([
                    filenames[x],
                    reference,
                    transcript,
                    cer / cer_ref,
                    wer / wer_ref
                ])

            total_wer += wer
            total_cer += cer
            num_tokens += wer_ref
            num_chars += cer_ref

            avg_total_wer += wer / wer_ref
            avg_total_cer += cer / cer_ref


        del out, out0, output_sizes, out_raw_cpu, out_softmax_cpu
        if (i + 1) % 5 == 0 or args.batch_size == 1:
            gc.collect()
            torch.cuda.empty_cache()

    if decoder is not None:
        wer_avg = float(total_wer) / num_tokens
        cer_avg = float(total_cer) / num_chars
        wer_avg2 = avg_total_wer / len(test_loader.dataset)
        cer_avg2 = avg_total_cer / len(test_loader.dataset)

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer_avg * 100, cer=cer_avg * 100))

        print('Alternative Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer_avg2 * 100, cer=cer_avg2 * 100))

    if args.output_path:
        import pickle
        with open(args.output_path, 'w') as f:
            f.write('\n'.join(processed_files))
