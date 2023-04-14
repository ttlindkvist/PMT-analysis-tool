def read_header_string(filename):
    header_dict = {}
    header_length = 0
    with open(filename) as file:
        line = ''
        while not ("[AndorNewton]" in line):
            line = next(file)
            header_length += 1
            if ':' in line:
                split_string = line.split(': ', 1)
                header_dict[split_string[0]] = split_string[1].strip()
            if '[Notes]' in line:
                line = next(file)
                header_length +=1
                comments = ''
                while not ('[Notes]' in line):
                    comments += line
                    line = next(file)
                    header_length += 1
                header_dict['[Notes]'] = comments.strip()
                
    return header_length, header_dict