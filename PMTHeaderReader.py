def read_header_string(filename):
    header_dict = {}
    header_length = 0
    with open(filename) as PMT_file:
        line = ''
        while not ("End of Header" in line):
            line = next(PMT_file)
            header_length += 1
            if not ("End of Header" in line) and len(line.split(': ',1)) > 1:
                split_string = line.split(': ', 1)
                header_dict[split_string[0]] = split_string[1].strip()
    return header_length, header_dict