import os 
import json
import argparse
import os.path as osp
import streamlit as st
from tools.visualization.constants import Constants

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('--result_folder', type=str,
                    help="Path to folder contains json result files")
parser.add_argument('-i', '--root_dir', type=str,
                    help="Path to root dir")
parser.add_argument('-s', '--split', type=str,
                    help="Specified split ['test', 'train']")
args = parser.parse_args()

CONSTANTS = Constants(args.root_dir)
queries_dict = json.load(open(CONSTANTS.QUERY_JSON[args.split], 'r'))
version_map = {
        json_name: osp.join(args.result_folder, json_name) for json_name in os.listdir(args.result_folder)
    }


srl_dict = None
if osp.isfile(CONSTANTS.SRL_JSON[args.split]):
    print("Loaded SRL prediction")
    srl_dict = json.load(open(CONSTANTS.SRL_JSON[args.split], 'r'))
    
color_dict = None
if osp.isfile(CONSTANTS.COLOR_JSON[args.split]):
    print("Loaded color prediction")
    color_dict = json.load(open(CONSTANTS.COLOR_JSON[args.split], 'r'))

vehicle_dict = None
if osp.isfile(CONSTANTS.VEHICLE_JSON[args.split]):
    print("Loaded vehicle prediction")
    vehicle_dict = json.load(open(CONSTANTS.VEHICLE_JSON[args.split], 'r'))

action_dict = None
if osp.isfile(CONSTANTS.STOP_TURN_JSON[args.split]):
    print("Loaded action prediction")
    action_dict = json.load(open(CONSTANTS.STOP_TURN_JSON[args.split], 'r'))


def load_prediction_meta(query_id):
    result = {
        'srl': None,
        'action': None,
        'color': None,
        'vehicle': None
    }
    if srl_dict:
        if query_id in srl_dict.keys():
            srl_track = srl_dict[query_id]
            main_subjects = [
                srl_track[srl_id]['main_subject'] 
                for srl_id in srl_track.keys()
            ]

            subjects = [] 
            colors = []
            actions = []

            for srl_track_id in list(srl_track.keys()):
                srl_dicts = srl_track[srl_track_id]['srl']
                for srl_dct in srl_dicts:
                    if len(srl_dct['subject_color']) > 0:
                        sbj_color = srl_dct['subject_color'][0]['color']
                    else:
                        sbj_color = None
                    sbj_action = srl_dct['action']
                    aux_sbj = srl_dct['subject']
                    subjects.append(aux_sbj)
                    colors.append(sbj_color)
                    actions.append(sbj_action)
            
            result['srl'] = {
                'main_sbj': main_subjects,
                'colors': colors,
                'actions': actions,
                'sub_sbj': subjects
            }

    if color_dict:
        if query_id in color_dict.keys():
            colors = color_dict[query_id]
            result['color'] = colors

    if vehicle_dict:
        if query_id in vehicle_dict.keys():
            vehicles = vehicle_dict[query_id]
            result['vehicle'] = vehicles
    
    if action_dict:
        actions = []
        for action_key in action_dict.keys():
            if query_id in action_dict[action_key]:
                actions.append(action_key)
        result['action'] = actions

    return result

def main(args):
    st.set_page_config(layout="wide")
    st.title('Traffic Video Event Retrieval via Text Query')

    list_versions = sorted(list(version_map.keys()))

    # Choose result version
    st.sidebar.subheader("Choose version")
    version = st.sidebar.radio(label="", options=list_versions)
    result_dict = json.load(open(version_map[version], 'r'))

    # Choose top k to retrieve
    top_to_show = st.sidebar.slider(
        'Show top-? results', 3, 30, 15)

    # Choose query id
    list_qids = list(result_dict.keys())
    display = [] 
    for qid in list_qids:
        display.append(f'{qid}')

    choose_qid = st.selectbox(f"Choose query", options=display)
    query_dict = queries_dict[choose_qid]
    list_vid_ids = result_dict[choose_qid]
    list_caps =  query_dict['nl'] 
    list_other_caps = query_dict['nl_other_views']

    if args.split == 'train':
        col1, col2 = st.columns([1, 2])

        # Write out query captions
        col1.markdown("### Query captions")
        col1.markdown("#### Main views")
        for cap in list_caps:
            col1.write(cap)
        col1.markdown("#### Other views")
        for cap in list_other_caps:
            col1.write(cap)
            
        # Visualize video
        col2.markdown("### Target video")
        video_name = f'{choose_qid}.mp4'
        video_path = osp.join(
            CONSTANTS.VIDEO_DIR[args.split], video_name)
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        col2.video(video_bytes)
        col2.text(video_name)

    else:
        # Write out query captions
        st.markdown("### Query captions")
        st.markdown("#### Main views")
        for cap in list_caps:
            st.write(cap)
        st.markdown("#### Other views")
        for cap in list_other_caps:
            st.write(cap)

    meta_prediction = load_prediction_meta(choose_qid)

    col1, col2, col3, col4 = st.columns([1,1,1,1])

    srl_pred_dict = meta_prediction['srl']
    col1.markdown("#### SRL Extraction")
    srl_cap = visualize_srl(srl_pred_dict)
    col1.markdown(
        srl_cap, unsafe_allow_html=True
    )
    
    col2.markdown("#### Predicted colors")
    pred_colors = meta_prediction['color']
    cap = visualize_color(pred_colors)
    col2.write(cap)

    col3.markdown("#### Predicted vehicles")
    pred_vehicles = meta_prediction['vehicle']
    cap = visualize_vehicle(pred_vehicles)
    col3.write(cap)

    col4.markdown("#### Predicted actions")
    pred_actions = meta_prediction['action']
    cap = visualize_actions(pred_actions)
    col4.write(cap)

    COLUMNS = 3
    ROWS = max(min(len(list_vid_ids), top_to_show) // COLUMNS, 1)

    # View text
    captions_list = []
    if args.split == 'train':
        # load query texts
        track_json = json.load(open(CONSTANTS.TRACKS_JSON[args.split], 'r'))
        captions_list = [track_json[i]["nl"]+['-------------'] + track_json[i]['nl_other_views'] for i in list_vid_ids]



    # Show retrieved video results
    st.markdown("### Retrieval results")
    if st.button('Search'):
        for r in range(ROWS):
            cols = st.columns(COLUMNS)
            for c in range(COLUMNS):
                vid_order = 3*r + c
                video_name = f'{list_vid_ids[vid_order]}.mp4'
                video_path = osp.join(CONSTANTS.VIDEO_DIR[args.split], video_name)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                cols[c].video(video_bytes)
                cols[c].text(f'{vid_order+1}. {video_name}')
                if len(captions_list) > 0:
                    if choose_qid == list_vid_ids[vid_order]:
                        captions = '<br>'.join(captions_list[vid_order])
                        new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 16px;">{captions}</p>'
                        cols[c].markdown(new_title, unsafe_allow_html=True)
                    else:
                        captions = '\n'.join(captions_list[vid_order])
                        cols[c].text(captions)
                
                # Load predicted metadata
                track_meta_info = load_prediction_meta(list_vid_ids[vid_order])
                track_meta_cap = visualize_all_pred(track_meta_info)
                cols[c].markdown(track_meta_cap, unsafe_allow_html=True)


def visualize_all_pred(track_meta_dict):
    srl_text = visualize_srl(track_meta_dict['srl'])
    color_text = visualize_color(track_meta_dict['color'])
    veh_text = visualize_vehicle(track_meta_dict['vehicle'])
    action_text = visualize_actions(track_meta_dict['action'])

    final_text = "<br>".join([
        "srl: <br>" + srl_text, 
        "color: " + color_text, 
        "vehicle: " + veh_text, 
        "action: " + action_text])
    return final_text

def visualize_srl(srl_pred_dict):
    # SRL dictionary
    if srl_pred_dict is not None:
        main_subjects = srl_pred_dict['main_sbj']
        colors = srl_pred_dict['colors']
        subjects = srl_pred_dict['sub_sbj']
        actions = srl_pred_dict['actions']

        main_title = []
        for main_sbj in main_subjects:
            sub_title = f'<span style="font-family:sans-serif; color:Black; font-size: 16px;">{main_sbj}</span>'
            main_title.append(sub_title)
        main_title = ', '.join(main_title)

        final_title = []
        for (sbj, clr, act) in zip(subjects, colors, actions):
            sub_title = f'<span style="font-family:sans-serif; color:Green; font-size: 16px;">{sbj}</span>'
            clr_title = f'<span style="font-family:sans-serif; color:Brown; font-size: 16px;">{clr}</span>'
            act_title = f'<span style="font-family:sans-serif; color:Blue; font-size: 16px;">{act}</span>'
            final_title.append(', '.join([sub_title, clr_title, act_title]))

        return main_title +  '<br>' + '<br>'.join(final_title)
    else:
        return "ID not found in srl json"

def visualize_color(color_pred):
    if color_pred is not None:
        color_pred = [CONSTANTS.COLOR_MAPPING[i] for i,v in enumerate(color_pred) if v]
        cap = ', '.join(color_pred)
    else:
        cap = "ID not found in color json"
    return cap

def visualize_vehicle(vehicle_pred):
    if vehicle_pred is not None:
        vehicle_pred = [CONSTANTS.VEHICLE_MAPPING[i] for i, v in enumerate(vehicle_pred) if v]
        cap = ', '.join(vehicle_pred)
    else:
        cap = "ID not found in vehicle json"
    return cap

def visualize_actions(actions):
    if len(actions) > 0:
        return ', '.join(actions)
    else:
        return "None" 

if __name__ == '__main__':
    main(args)