import scrapy


#SCRAPY SPIDER
class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = ['https://duma.mos.ru/ru/60/conference?date%5Bfrom%5D=0&date%5Bto%5D=0&sort=date-desc&page=1#results']
    page_number = 1
    
    def parse(self, response):
        text_urls = response.css('td:nth-child(4) .link-img').css('::attr(href)').getall()
        yield {
            'urls':text_urls,
            'page':QuotesSpider.page_number
        }

        column = response.css('tr+ tr td:nth-child(5)')
        for i in range(1,20):
            video_urls = column.css('td a:contains("Трансляция {} ")'.format(i)).css('::attr(href)').getall() 
            #video_urls = response.css('tr:nth-child({}) td:nth-child(5)'.format(i)).css('::attr(href)').getall()[2:]
            if not video_urls:
                continue
            for video in video_urls:
                video_link = response.urljoin(video)
                request = scrapy.Request(video_link, callback=self.parse_video)
                request.meta['page'] = QuotesSpider.page_number
                request.meta['block'] = i
                yield request
        
        QuotesSpider.page_number+=1
        next_page = 'https://duma.mos.ru/ru/60/conference?date%5Bfrom%5D=0&date%5Bto%5D=0&sort=date-desc&page=' + str(QuotesSpider.page_number) + '#results'
        
        if QuotesSpider.page_number < 31:
            yield response.follow(next_page, callback=self.parse)
    
    def parse_video(self, response):
        
        page = response.meta.get('page')
        block = response.meta.get('block')
        video_download = response.css('video').css('::attr(src)').get()
        yield {
            'video':video_download,
            'page':page,
            'block':block
        }

        

            