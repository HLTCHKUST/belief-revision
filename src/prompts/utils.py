import re


def get_final_answer(full_answer, keyword = 'final answer'):
    snip = full_answer[full_answer.lower().rfind(keyword)+len(keyword):]
    # if there's any alphabet inside snip
    if sum([char.isalpha() for char in snip])>0:
        first_alpha = snip.find(next(filter(str.isalpha, snip)))
        next_index = first_alpha+1
        if next_index < len(snip):
            if snip[next_index].isalpha():
                return ''
            else:
                return re.sub(r'\W+', '', snip)[0].lower()
        else:
            return re.sub(r'\W+', '', snip)[0].lower()
    else:
        return ''