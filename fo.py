def format_output(string):
    array = string.split()
    for i in range(len(array)):
        if array[i].lower() == '<|assistant|>':
            new_array = array[i+1:]
            break

    result = " ".join(new_array)
    return result