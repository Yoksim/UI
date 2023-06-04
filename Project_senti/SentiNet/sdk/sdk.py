import requests, xlrd, xlwt

LANG = 'en'
APIKEY = 'gWgp0lKQ4OrE'
APIURL = 'https://sentic.net/api/' + LANG + '/' + APIKEY + '.py?text='
FILENAME = 'data'

wb = xlrd.open_workbook(FILENAME + '.xls')
sheet = wb.sheet_by_index(0)
new_wb = xlwt.Workbook()
new_sheet = new_wb.add_sheet('labeled')

count = 0
for row in range(sheet.nrows):
    text = sheet.cell_value(row, 0)
    for c in [';', '&', '#', '{', '}']: text = text.replace(c, ':')
    label = str(requests.get(APIURL + text).content)[2:-3]
    new_sheet.write(count, 0, text)
    if label == 'NEGATIVE': 
        new_sheet.write(count, 1, label, xlwt.easyxf('font: color red; align: horiz center'))
        print(text + ': \033[91m' + label + '\033[0;0m')
    elif label == 'POSITIVE':
        new_sheet.write(count, 1, label, xlwt.easyxf('font: color green; align: horiz center'))
        print(text + ': \033[92m' + label + '\033[0;0m')
    else:
        new_sheet.write(count, 1, label, xlwt.easyxf('align: horiz center'))
        print(text + ': ' + label)
    count += 1
    
new_wb.save(FILENAME + '_labeled.xls')