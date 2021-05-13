import os

# list of author names, each author must correspond to a subdirectory in texts/
AUTHORS = ['Chapman', 'Cowper', 'Dryden', 'Pope']

def read_data(authors=AUTHORS):
    '''
    Read the data for each author in the `authors` list which also
    appears in the `AUTHORS` list.

    Returns a dict in the form:
    {
        author_name: {
            'iliad': iliad,
            'poetry': [text, text, ...]
        },
        ...
    }

    This function assumes it's being called in the root directory of the entire project,
    otherwise the relative paths fail.
    '''
    data = {}

    # for each author
    for auth in [x for x in AUTHORS if x in authors]:
        # list all files in author's directory
        files = os.listdir(f'texts/{auth}')
        # compile all the texts
        texts = {}
        for file in files:
            contents = open(f'texts/{auth}/{file}', 'r', encoding='utf-8').read().strip()
            if 'iliad' in file:
                texts['iliad'] = contents
            else:
                if 'poetry' not in texts: texts['poetry'] = []
                texts['poetry'].extend(contents.split('\n\n'))
        # store in dict
        data[auth] = texts

    return data
