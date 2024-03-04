from selenium import webdriver
from filemanager import FileManager

class WebSpider:
    def __init__(self, file_manager = None, scrape_function = None,name=''):
        
        self.file_manager = file_manager
        self.scrape = scrape_function
        self.name = name if name != None else id(self)

    def get_from_file(self,fn):
        
        item = self.file_manager.dequeue_from_file(fn)
        self.file_manager.write_file(self.name+'_doing_'+item+'.txt')
        return item

    def add_to_file(self,text,fn):

        self.file_manager.enqueue_to_file(text,fn)
        self.file_manager.remove_file(self.name+'_doing_'+text+'.txt')

    def push_to_file(self,text,fn):

        self.file_manager.push_to_file(text,fn)
        self.file_manager.remove_file(self.name+'_doing_'+text+'.txt')

    # def write_current(text):
    #     self.file_manager.write_file(self.name+'_doing_'+text+'.txt')

    # def remove_current(text):
    #     self.file_manager.remove_file(self.name+'_doing_'+text+'.txt')
        
    def set_scrape_function(self,scrape_function):

        self.scrape_website = scrape_function


    def close(self):
        # Close the browser window
        pass

# Example usage:
# spider = WebSpider('https://example.com')
# spider.scrape_website()
# spider.close()
