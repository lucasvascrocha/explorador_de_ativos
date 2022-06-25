from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
#from tqdm.notebook import tqdm
#from selenium.webdriver.chrome.options import Options
import sys
import requests
import http.cookiejar
import re
import urllib.request
from lxml.html import fragment_fromstring
from collections import OrderedDict
import urllib.parse
import pandas as pd





def get_data(*args, **kwargs):
    class AppURLopener(urllib.request.FancyURLopener):
      version = "Mozilla/5.0"

    opener = AppURLopener()
    response = opener.open('http://httpbin.org/user-agent')

    url = 'http://www.fundamentus.com.br/resultado.php'
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'),
                        ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')                                                             
                        ]

    #opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201'),
    #                     ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')]

    # Aqui estão os parâmetros de busca das ações
    # Estão em branco para que retorne todas as disponíveis
    data = {'pl_min':'','pl_max':'','pvp_min':'','pvp_max' :'','psr_min':'','psr_max':'','divy_min':'','divy_max':'',            'pativos_min':'','pativos_max':'','pcapgiro_min':'','pcapgiro_max':'','pebit_min':'','pebit_max':'', 'fgrah_min':'',
            'fgrah_max':'', 'firma_ebit_min':'', 'firma_ebit_max':'','margemebit_min':'','margemebit_max':'',            'margemliq_min':'','margemliq_max':'', 'liqcorr_min':'','liqcorr_max':'','roic_min':'','roic_max':'','roe_min':'',            'roe_max':'','liq_min':'','liq_max':'','patrim_min':'','patrim_max':'','divbruta_min':'','divbruta_max':'',         'tx_cresc_rec_min':'','tx_cresc_rec_max':'','setor':'','negociada':'ON','ordem':'1','x':'28','y':'16'}

    with opener.open(url, urllib.parse.urlencode(data).encode('UTF-8')) as link:
        content = link.read().decode('ISO-8859-1')

    pattern = re.compile('<table id="resultado".*</table>', re.DOTALL)
    reg = re.findall(pattern, content)[0]
    page = fragment_fromstring(reg)
    lista = OrderedDict()


    stocks = page.xpath('tbody')[0].findall("tr")

    todos = []
    for i in range(0, len(stocks)):
        lista[i] = {
            stocks[i].getchildren()[0][0].getchildren()[0].text: {
                'cotacao': stocks[i].getchildren()[1].text,
               'P/L': stocks[i].getchildren()[2].text,
               'P/VP': stocks[i].getchildren()[3].text,
               'PSR': stocks[i].getchildren()[4].text,
               'DY': stocks[i].getchildren()[5].text,
               'P/Ativo': stocks[i].getchildren()[6].text,
               'P/Cap.Giro': stocks[i].getchildren()[7].text,
               'P/EBIT': stocks[i].getchildren()[8].text,
               'P/Ativ.Circ.Liq.': stocks[i].getchildren()[9].text,
               'EV/EBIT': stocks[i].getchildren()[10].text,
               'EBITDA': stocks[i].getchildren()[11].text,
               'Mrg. Ebit': stocks[i].getchildren()[12].text,
               'Mrg.Liq.': stocks[i].getchildren()[13].text,
               'Liq.Corr.': stocks[i].getchildren()[14].text,
               'ROIC': stocks[i].getchildren()[15].text,
               'ROE': stocks[i].getchildren()[16].text,
               'Liq.2m.': stocks[i].getchildren()[17].text,
               'Pat.Liq': stocks[i].getchildren()[18].text,
               'Div.Brut/Pat.': stocks[i].getchildren()[19].text,
               'Cresc.5a': stocks[i].getchildren()[20].text
               }
            }

    return lista

def get_specific_data(stock):
    class AppURLopener(urllib.request.FancyURLopener):
      version = "Mozilla/5.0"

    opener = AppURLopener()
    response = opener.open('http://httpbin.org/user-agent')

    url = "http://www.fundamentus.com.br/detalhes.php?papel=" + stock
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'),
                        ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')                                                             
                        ]
    
    
    
    
    
    #opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201'),
    #                     ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')]
    
    # Get data from site
    link = opener.open(url, urllib.parse.urlencode({}).encode('UTF-8'))
    content = link.read().decode('ISO-8859-1')

    # Get all table instances
    pattern = re.compile('<table class="w728">.*</table>', re.DOTALL)
    reg = re.findall(pattern, content)[0]
    reg = "<div>" + reg + "</div>"
    page = fragment_fromstring(reg)
    all_data = {}

    # There is 5 tables with tr, I will get all trs
    all_trs = []
    all_tables = page.xpath("table")

    for i in range(0, len(all_tables)):
        all_trs = all_trs + all_tables[i].findall("tr")

    # Run through all the trs and get the label and the
    # data for each line
    for tr_index in range(0, len(all_trs)):
        tr = all_trs[tr_index]
        # Get into td
        all_tds = tr.getchildren()
        for td_index in range(0, len(all_tds)):
            td = all_tds[td_index]

            label = ""
            data = ""

            # The page has tds with contents and some 
            # other with not
            if (td.get("class").find("label") != -1):
                # We have a label
                for span in td.getchildren():
                    if (span.get("class").find("txt") != -1):
                        label = span.text

                # If we did find a label we have to look 
                # for a value 
                if (label and len(label) > 0):
                    next_td = all_tds[td_index + 1]

                    if (next_td.get("class").find("data") != -1):
                        # We have a data
                        for span in next_td.getchildren():
                            if (span.get("class").find("txt") != -1):
                                if (span.text):
                                    data = span.text
                                else:
                                    # If it is a link
                                    span_children = span.getchildren()
                                    if (span_children and len(span_children) > 0):
                                        data = span_children[0].text

                                # Include into dict
                                all_data[label] = data

                                # Erase it
                                label = ""
                                data = ""

    return all_data




def coletar_scrap():
    sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
    URL = 'https://statusinvest.com.br/acoes/busca-avancada'
    #output = 'busca-avancada.csv'

    #if path.exists(output):
    #    os.remove(output)

    #chrome_options = Options()
    #chrome_options.binary_location = GOOGLE_CHROME_BIN
    #chrome_options.add_argument('--disable-gpu')
    #chrome_options.add_argument('--no-sandbox')
    #driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)

    #driver = webdriver.Chrome('chromedriver/chromedriver.exe')

    #chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument('--headless')
    #chrome_options.add_argument('--no-sandbox')
    #chrome_options.add_argument('--disable-dev-shm-usage')
    #driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)

    gChromeOptions = webdriver.ChromeOptions()
    gChromeOptions.add_argument("window-size=1920x1480")
    gChromeOptions.add_argument("disable-dev-shm-usage")
    driver = webdriver.Chrome(
        chrome_options=gChromeOptions, executable_path=ChromeDriverManager().install()
    )
    
    #driver = webdriver.Chrome(ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install())
    #driver.get(URL)
    driver.get(URL)
    sleep(5)

    button_buscar = driver.find_element_by_xpath('//div/button[contains(@class,"find")]')

    button_buscar.click()
    sleep(5)

    button_skip = driver.find_element_by_xpath('//div/button[contains(@class,"btn-close")]')

    button_skip.click()
    sleep(5)

    button_download = driver.find_element_by_xpath('//div/a[contains(@class, "btn-download")]')
    button_download.click()
    sleep(1)
    
    #if path.exists(output):
               

    df = pd.read_csv('busca-avancada.csv', sep=';', decimal=',', thousands='.')
    driver.close()
    return df


def scrap_fundamentus():
    url  = 'http://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    result = requests.get(url, headers=headers)
    df = pd.read_html(result.content)[0]
    
    return df