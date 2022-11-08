import os.path as osp
import re

import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from .color_helper import ColorHelper
from .data_helper import DataHelper
from .srl_helper import SRLHelper


class SRL(object):
    def __init__(self, path=None):
        textual_module_dir = osp.dirname(osp.abspath(__file__))
        extract_module_dir = osp.dirname(textual_module_dir)
        self.srl_helper = SRLHelper(
            osp.join(extract_module_dir, "configs/srl_helper/srl_helper.json"),
            get_predictor=True,
        )
        if path is not None:
            data_helper = DataHelper()
            self.data = data_helper.load_file(path)
            self.data = data_helper.convert_json_train(self.data)

        self.color_helper = ColorHelper()
        self.spacy_model = spacy.load("en_core_web_sm")
        self.wnl = WordNetLemmatizer()

    def extract_query(self, query):
        list_pos_parts = []
        extract_text = self.srl_helper.predictor.run_sentence(query)["verbs"]
        if not extract_text:
            return list_pos_parts
        try:
            for text in extract_text:
                sentence = text["description"]
                pos_part = re.findall(r"\[(.*?)\]", sentence)
                pos_part = [
                    self.srl_helper.clean_query_after_out(sub_pos_part)
                    for sub_pos_part in pos_part
                ]
                list_pos_parts.append(pos_part)
        except:
            list_pos_parts = []
        return list_pos_parts

    def extract_sub_pos_part(self, sub_pos_part, indicator, distance):
        index = sub_pos_part.find(indicator)
        return sub_pos_part[:index], sub_pos_part[index + distance :]

    def get_subject(self, list_pos_parts, query):
        verb_indicating = "V:"
        i = 0
        flag_subject = False
        subject_phrase = None
        while i < len(list_pos_parts) and not flag_subject:
            pos_part = list_pos_parts[i]
            if pos_part[0].find(verb_indicating) >= 0:
                i += 1
            else:
                subject_phrase = self.get_subject_phrase(pos_part)
                flag_subject = True
        if i == len(list_pos_parts) or subject_phrase is None:
            subject = self.get_subject_from_query(query)
            return subject
        subject = self.get_subject_from_phrase(subject_phrase)
        if subject is not None:
            return subject
        return self.get_subject_from_query(query)

    def get_subject_phrase(self, pos_part):
        subject_phrase = None
        sub_pos_part = self.extract_sub_pos_part(
            sub_pos_part=pos_part[0], indicator=":", distance=2
        )
        if sub_pos_part and re.match(r"ARG[0-4]{1}$", sub_pos_part[0]) is not None:
            subject_phrase = sub_pos_part[1]
        return subject_phrase

    def get_subject_from_phrase(self, phrase):
        list_vehicle = self.srl_helper.list_vehicle
        for key_level in list_vehicle:
            for vehicle in list_vehicle[key_level]:
                if phrase.find(vehicle) >= 0:
                    return vehicle
        return None

    def get_subject_from_query(self, query):
        list_vehicle = self.srl_helper.list_vehicle
        tokens = nltk.word_tokenize(query)
        for token in tokens:
            for key_level in list_vehicle:
                if token in list_vehicle[key_level]:
                    return token
        return None

    def get_action_feature(self, pos_part):
        content_indicator = ":"
        distance = 2
        verb_indicator = "V:"

        verb_pos = 0
        while verb_pos < len(pos_part) and pos_part[verb_pos].find(verb_indicator) < 0:
            verb_pos += 1

        if verb_pos == len(pos_part) or verb_pos == 0:
            return None

        data = {
            "subject": None,
            "subject_color": None,
            "action": None,
            "arg_1": None,
            "arg_2": None,
            "arg_3": None,
            "arg_4": None,
            "argm_mnr": None,
            "argm_dir": None,
            "argm_loc": None,
            "argm_tmp": None,
            "argm_adv": None,
        }

        # get subject
        flag = -1
        for i in range(verb_pos):
            subject = self.extract_sub_pos_part(
                pos_part[i], content_indicator, distance
            )
            list_vehicle = self.srl_helper.list_vehicle
            if re.match(r"ARG[0-4]{1}$", subject[0]):
                for key_level in list_vehicle:
                    for vehicle in list_vehicle[key_level]:
                        if subject[1].find(vehicle) >= 0:
                            data["subject"] = subject[1]
                            flag = i
        if data["subject"] is None:
            return None

        # get action feature
        for i in range(len(pos_part)):
            if i == flag:
                continue
            feature = self.extract_sub_pos_part(
                pos_part[i], content_indicator, distance
            )
            semantic_indicator = feature[0]
            for key_indicator in self.srl_helper.semantic_key_converter:
                if (
                    self.srl_helper.semantic_key_converter[key_indicator]
                    == semantic_indicator
                ):
                    data[key_indicator] = feature[1]
                    if key_indicator == "action":
                        data[key_indicator] = self.wnl.lemmatize(feature[1], "v")
        # get subject color:
        data["subject_color"] = self.color_helper.extract_color(data["subject"])

        return data

    def extract_noun_phrase(self, cap):
        """Extract subject and color in noun phrase. Ex: a black sedan.
        Args:
            cap ([type]): [description]
            spacy_model ([type]): [description]
            veh_vocab (list): [description]
        Returns:
            result: {'S': sedan, 'colors': [black] }
        """
        spacy_model = self.spacy_model
        veh_vocab = self.srl_helper.vehicle_vocab

        tokens = spacy_model(cap)
        result = {"subject": None, "colors": None}
        # Get subject

        for phrase in tokens.noun_chunks:
            words = phrase.text.split(" ")
            veh, veh_id = None, None

            for i, word in enumerate(words):
                if word in veh_vocab:
                    veh = word
                    veh_id = None
                    break

            if veh is None:
                continue

            cols = self.color_helper.extract_color(" ".join(words[:veh_id]))
            result["subject"] = veh
            result["colors"] = cols
            break

        return result

    def do_extraction(self, data):
        ans = {}
        list_null_subject = []
        key_id = 0
        count_error = 0

        for key in tqdm(data):
            ans_key = {}
            key_id += 1
            query_number = 0
            # for view in ["nl", "nl_other_views"]:
            queries = data[key]  # [view]
            for query in queries:
                ans_query = {"caption": query, "main_subject": None, "srl": []}
                query_number += 1
                query_id = f"{key_id}_{query_number}"
                query = self.srl_helper.clean_query_before_inp(query)
                ans_query["cleaned_caption"] = query
                list_pos_parts = self.extract_query(query)
                if not list_pos_parts:
                    count_error += 1
                    print(f"({count_error}) {key_id}_{query_number}: {query}")
                    ans_query["is_extractable"] = False
                    extract_noun_phrase = self.extract_noun_phrase(query)
                    if extract_noun_phrase["subject"] is None:
                        list_null_subject.append((key, query_id))
                        ans_query["is_original_subject"] = False
                    else:
                        ans_query["main_subject"] = extract_noun_phrase["subject"]
                        ans_query["is_original_subject"] = True
                        ans_query["colors"] = extract_noun_phrase["colors"]

                else:
                    ans_query["is_extractable"] = True
                    subject = self.get_subject(list_pos_parts, query)

                    if subject is None:
                        list_null_subject.append((key, query_id))
                        ans_query["is_original_subject"] = False
                    else:
                        ans_query["main_subject"] = subject
                        ans_query["is_original_subject"] = True
                        ans_query["srl"] = self.srl(list_pos_parts)

                ans_key[query_id] = ans_query
            ans[key] = ans_key
        return ans, list_null_subject

    def srl(self, list_pos_parts):
        ans = []
        i = 0
        verb_indicator = "V:"
        while i < len(list_pos_parts):
            pos_part = list_pos_parts[i]
            if pos_part[0].find(verb_indicator) >= 0:
                i += 1
                continue
            action_feature = self.get_action_feature(pos_part)
            if action_feature is not None:
                ans.append(action_feature)
            i += 1
        return ans

    def get_subject_from_same_group_query(self, list_subject):
        if len(list_subject) == 1:
            return list_subject[0]
        else:
            vehicle_1 = list_subject[0]
            vehicle_2 = list_subject[1]
            if vehicle_1 == vehicle_2:
                return vehicle_1

            vehicle_lv_1 = 1
            vehicle_lv_2 = 1
            for key_level in self.srl_helper.list_vehicle:
                if vehicle_1 in self.srl_helper.list_vehicle[key_level]:
                    vehicle_lv_1 = int(key_level)
                if vehicle_2 in self.srl_helper.list_vehicle[key_level]:
                    vehicle_lv_2 = int(key_level)

            if vehicle_lv_1 <= vehicle_lv_2:
                return vehicle_1
            else:
                return vehicle_2

    def get_subject_from_null_data(self, ans, list_null_subject):
        for item in tqdm(list_null_subject):
            key_group, key_query = item[0], item[1]
            list_subject = []
            for query in ans[key_group]:
                if query == key_query:
                    continue
                main_subject = ans[key_group][query]["main_subject"]
                if main_subject is not None:
                    list_subject.append(main_subject)
            subject = self.get_subject_from_same_group_query(list_subject)
            ans[key_group][key_query]["main_subject"] = subject
        return ans

    def compare_subject_object_from_all_queries(self, ans):
        for key_group_query in ans:
            for key_query in ans[key_group_query]:
                ans_query = ans[key_group_query][key_query]
                ans_query = self.compare_subject_object_from_single_query(
                    ans_query, ans_query["main_subject"]
                )
                ans[key_group_query][key_query] = ans_query
        return ans

    def compare_subject_object_from_single_query(self, ans_query, subject):
        ans_srl = ans_query["srl"]
        if not ans_srl:
            return ans_query
        for i in range(len(ans_srl)):
            subject_srl_item = ans_srl[i]["subject"]
            key_item = [subject, "it "]
            skip_item = ["others", "another", "other", "no", "different"]
            skip_item_query = ["there ", "there "]
            if (
                any(key in subject_srl_item for key in key_item)
                and (
                    not any(
                        ans_query["cleaned_caption"].find(skip) == 0
                        for skip in skip_item_query
                    )
                )
                and (not any(subject_srl_item.find(skip) == 0 for skip in skip_item))
            ):
                ans_query["srl"][i]["is_main_subject"] = True
            else:
                ans_query["srl"][i]["is_main_subject"] = False
        return ans_query

    def extract_data(self, data):
        self.ans, self.list_null_subject = self.do_extraction(data)
        self.ans = self.get_subject_from_null_data(self.ans, self.list_null_subject)
        self.ans = self.compare_subject_object_from_all_queries(self.ans)
        return self.ans
