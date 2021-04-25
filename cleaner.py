import re
#import nltk

#txt_f = input('File name: ')
#out_f = input('Output file: ')

txt_f = 'in.txt'
out_f = 'out.txt'

with open(txt_f, 'r', encoding='utf-8') as f:
    text = f.read()

# comment/uncomment as needed for your particular file
text = re.sub('\[.*?\]', '', text, flags=re.DOTALL) # remove all notes, which are contained in brackets
text = re.sub('\*[\s]*[\w]*\*', '', text)           # remove footnotes contained between *
text = re.sub('[0-9]+\.*', '', text)                # remove all line numbers
text = re.sub('[IVXLCM]+\.', '', text)              # remove all roman numerals (stanza markers)
text = re.sub('\n[\r\t\f\v ]+', '\n', text)         # remove all extraneous whitespace between stanzas and at line beginnings
text = re.sub('[ \t]*\d+[ \t]*\n', '\n', text)      # remove line numbers that appear at the end of lines
text = re.sub('\n[ \t]*\d+[ \t]*', '\n', text)      # remove line numbers that appear at the beginning of lines

with open(out_f, 'w', encoding='UTF-8') as f:
    f.write(text)
