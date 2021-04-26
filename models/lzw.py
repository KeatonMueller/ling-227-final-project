def compression_size(text):
    "calculate the number of bits the lzw algorithm would compress `text` into"
    
    # represent string table as dict of form { (prefix, char): code }
    table = {}

    # initialize string table with all 1-character strings
    code = 1
    for char in set(text):
        table[(0, char)] = code
        code += 1

    prefix, char = 0, ''
    total_bits = 0

    for i in range(len(text)):
        char = text[i]
        # update prefix if (prefix, char) is in string table
        if (prefix, char) in table:
            prefix = table[(prefix, char)]
        # otherwise output prefix and update string table
        else:
            total_bits += prefix.bit_length()
            table[(prefix, char)] = code
            code += 1
            prefix = table[(0, char)]

    if prefix != 0:
        total_bits += prefix.bit_length()

    return total_bits