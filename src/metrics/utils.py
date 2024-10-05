import editdistance

def calc_wer(target_text: str, pred_text: str):
    if len(target_text) == 0:
        if pred_text != '':
            return 1.
        return 0.
    return editdistance.eval(target_text.split(), pred_text.split()) / len(target_text.split())
    

def calc_cer(target_text: str, pred_text: str):
    if len(target_text) == 0:
        if pred_text != '':
            return 1.
        return 0.
    return editdistance.eval(target_text, pred_text) / len(target_text)
