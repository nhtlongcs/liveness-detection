import spacy
import re

from .text_utils import (
    get_color, get_args_from_srl_sample, extract_noun_phrase
)
from .constant import (
    FOLLOW, FOLLOW_BY,
    OPPOSITE,
    HAS_FOLLOW, NO_FOLLOW, NO_CONCLUSION
)
from external.extraction.paths import VEHICLE_VOCAB, ACTION_VOCAB, VEHICLE_VOCAB_OBJ

nlp_model = spacy.load('en_core_web_sm')

class Vehicle(object):
    def __init__(self, vehicle: str, colors: list, get_none=False):
        self.vehicle = vehicle 
        self.colors = []
        self.combines = []    
        for col_pair in colors:
            color, adv = col_pair['color'], col_pair['adv']
            self.colors.append(color)
            if adv:
                self.combines.append(f'{adv}_{color}')
            else:
                self.combines.append(color)
        
        if get_none and not self.combines:
            self.combines.append(None)

    def __str__(self):
        return f'veh={self.vehicle}, col={self.combines}'

class Caption(object):
    def __init__(self, cap_content: dict, cap_id: str):
        self.cap_id = cap_id
        self.__dict__.update(cap_content)
        self.sv_format, self.svo_format = [], []
        self._setup()
        pass

    def _extract_object(self, srl_content):
        """
        Args:
            srl_content ([type]): [description]
        Returns:
            [type]: [description]
        """
        args = get_args_from_srl_sample(srl_content)
        if args is None:
            return None
        
        list_objs = []
        main_object = None
        for arg in args:
            # tokens = self.nlp_model(arg)
            # tokens = nlp_model(arg)
            tokens = arg.split(' ')
            colors = get_color(tokens)
            for tok in tokens:
                if any(tok.find(veh) >= 0 for veh in VEHICLE_VOCAB_OBJ):
                # if tok in VEHICLE_VOCAB:
                # if ('NN' in tok.pos_) and (tok in VEHICLE_VOCAB):
                    # list_colors.append(colors)
                    main_object = Vehicle(vehicle=tok, colors=colors, get_none=True)
                    # list_objs.append(Vehicle(vehicle=tok, color=colors))
        
        return main_object
    
    def _find_subject_pronoun(self, srl_content):
        args = get_args_from_srl_sample(srl_content)
        if args is None:
            return False
        for arg in args:
            tokens = arg.split(' ')
            for tok in tokens:
                if tok == "it":
                    return True
        return False
    
    def _create_sv_sample(self, action):
        return {'S': self.subject, 'V': action}

    def _create_svo_sample(self, action, main_object):
        return {'S': self.subject, 'V': action, 'O': main_object}
    
    def _init_main_subject(self):
        all_colors, obj_colors = [], []
        flag = False
        for srl in self.srl:
            action = srl['action']
            if srl['is_main_subject']:
                flag = True
                obj_colors.extend(srl['subject_color'])
        
        if flag:
            self.subject = Vehicle(vehicle=self.main_subject, colors=obj_colors)
        else:
            self.subject = Vehicle(vehicle="", colors=[])
            self.main_subject = ""

    def _setup(self):
        # if len(self.srl) <= 1:
        #     self.is_svo = False 
        
        if len(self.srl) == 0:
            self.is_follow = NO_CONCLUSION
            extract_result = extract_noun_phrase(self.cleaned_caption, nlp_model, VEHICLE_VOCAB)
            if extract_result is not None:
                self.subject = Vehicle(vehicle=extract_result['S'], colors=extract_result['colors'])
            else:
                self.subject = Vehicle(vehicle=self.main_subject, colors=[])
            
        else:
            self._init_main_subject()
            self._extract_all_follow()
            self._extract_other_action()
        pass

    def _find_all_follow(self, cap):
        list_flb = [(cap.start(), FOLLOW_BY) for cap in re.finditer(FOLLOW_BY, cap)]
        while cap.find(FOLLOW_BY) >= 0:
            cap = cap.replace(FOLLOW_BY, " " * len(FOLLOW_BY))
        list_fl = [(cap.start(), FOLLOW) for cap in re.finditer(FOLLOW, cap)]
        list_flb.extend(list_fl)
        ans = sorted(list_flb, key=lambda tup: tup[0])
        return ans

    def _extract_all_follow(self):
        list_fl = self._find_all_follow(self.cleaned_caption)
        if not list_fl:
            self.is_follow = NO_CONCLUSION
            return
        self.is_follow = HAS_FOLLOW
        pos = 0
        for item in list_fl:
            action = item[1]
            
            while pos < len(self.srl) and self.srl[pos]['action'].find(FOLLOW) < 0:
                pos += 1
                continue
            if pos >= len(self.srl):
                return

            check_obj = self._find_subject_pronoun(self.srl[pos])
            if not self.srl[pos]['is_main_subject'] or check_obj:
                if self.srl[pos]['subject'].find("no") == 0:
                    self.is_follow = NO_FOLLOW
                    return
                else:
                    self.srl[pos]["arg_1"] = self.srl[pos]["subject"]
                    self.srl[pos]["subject"] = ""
                    self.srl[pos]["subject_color"] = []
                    action = OPPOSITE[action]
                pass

            obj = self._extract_object(self.srl[pos])
            self.srl[pos]['action'] = action
            pos += 1

            if obj is None:
                continue
            self.svo_format.append(self._create_svo_sample(action, obj))

    def _extract_other_action(self):
        for srl in self.srl:
            action = srl['action']

            # check whether follow action
            if action.find(FOLLOW) >= 0:
                continue
            if (not srl['is_main_subject']):
                continue

            obj = self._extract_object(srl)
            if (action in ACTION_VOCAB):
                self.sv_format.append(self._create_sv_sample(action))
            elif (obj is not None):
                self.svo_format.append(self._create_svo_sample(action, obj))