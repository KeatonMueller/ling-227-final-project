import os

# list of author names, each author must correspond to a subdirectory in texts/
AUTHORS = ['Chapman', 'Cowper', 'Dryden', 'Pope']

def read_data(authors=AUTHORS):
    '''
    Read the data for each author in the `authors` list which also
    appears in the `AUTHORS` list.

    Return all data for each author, in the form { author_name: [Iliad, text, text, ...] },
    with the Iliad always appearing in the first position in the list of texts.

    This function assumes it's being called in the root directory of the entire project,
    otherwise the relative paths fail.
    '''
    data = {}

    # for each author
    for auth in [x for x in AUTHORS if x in authors]:
        # list all files in author's directory
        files = os.listdir(f'texts/{auth}')
        # compile all the texts
        texts = []
        for file in files:
            contents = open(f'texts/{auth}/{file}', 'r').read().strip()
            # always put iliad in first position in list
            if 'iliad' in file:
                texts.insert(0, contents)
            else:
                texts.extend(contents.split('\n\n'))
        # store in dict
        data[auth] = texts

    return data
