import webcolors
from matplotlib.colors import is_color_like

def extract_noun_phrase(cap, spacy_model, veh_vocab: list):
    """Extract subject and color in noun phrase. Ex: a black sedan.
    Args:
        cap ([type]): [description]
        spacy_model ([type]): [description]
        veh_vocab (list): [description]
    Returns:
        result: {'S': sedan, 'colors': [black] }
    """
    tokens = spacy_model(cap)
    subject = None 
    result = None
    # Get subject
    
    for phrase in tokens.noun_chunks:
        words = phrase.text.split(' ')
        veh, veh_id = None, None
            
        for i, word in enumerate(words):
            if word in veh_vocab:
                veh = word
                veh_id = None 
                break
        
        if veh is None:
            continue 

        cols = get_color(words[:veh_id])
        result = {'S': veh, 'colors': cols}
        break
    
    return result
    
def refine_srl_args(arg: str):
    arg = arg.replace(' - ', '-')
    return arg

def check_color_adv(word: str):
    return word.lower() in ['light', 'dark']

def get_color(spacy_out: str):
    res = []
    for i, word in enumerate(spacy_out):
        if isinstance(word, str):
            text = word.lower()
        else:
            text = word.text.lower()
        if is_color_like(text):
            # if is_color_like(text):
            color_prop = {'color': text, 'adv': None}
            if i >= 1 and ((spacy_out[i-1] + text) in webcolors.CSS3_NAMES_TO_HEX 
                            or check_color_adv(spacy_out[i-1])
            ):
                color_prop['adv'] = spacy_out[i-1]
            
            res.append(color_prop)
    
    return res

def get_args_from_srl_sample(srl_content: dict):
    arg_keys = ['arg_1', 'arg_2', 'arg_3', 'arg_4', 'argm_loc']
    list_args = []
    for arg_key in arg_keys:
        if srl_content[arg_key] is not None:
            list_args.append(refine_srl_args(srl_content[arg_key]))
    
    if len(list_args) == 0:
        return None

    return list_args
