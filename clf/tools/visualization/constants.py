class Constants:
    def __init__(self, root_path) -> None:
        
        ## Roots
        self.AIC22_ROOT = root_path
        self.AIC22_ORI_ROOT = f"{self.AIC22_ROOT}/AIC22_Track2_NL_Retrieval"
        self.AIC22_META_ROOT = f"{self.AIC22_ROOT}/meta"

        self.TRACKS_JSON = {
            'test': f"{self.AIC22_META_ROOT}/test_tracks.json",
            'train': f"{self.AIC22_META_ROOT}/train_tracks.json",
        }

        # Queries data
        self.QUERY_JSON = {
            'test': f"{self.AIC22_META_ROOT}/test_queries.json",
            'train': self.TRACKS_JSON['train'],
        }

        ## Extracted frames from videos, generated from `tools/extract_vdo2frms_AIC.sh`
        self.EXTRACTED_FRAMES_DIR = f"{self.AIC22_META_ROOT}/extracted_frames"

        ## Video track, generated from `tools/visualization/video_gen` 
        self.VIDEO_DIR = {
            'test': f"{self.AIC22_META_ROOT}/track_visualization/relation/test-convert",
            'train': f"{self.AIC22_META_ROOT}/track_visualization/relation/train-val-convert",
        }
        
        # METADATA PREDICTIONS

        ### Generated from `tools/preproc_srl.sh`
        self.SRL_JSON = {
            'train': f'{self.AIC22_META_ROOT}/srl_train_tracks.json',
            'test': f'{self.AIC22_META_ROOT}/srl_test_queries.json',
        }
        
        ### Generated from `tools/preproc_relation.sh`
        self.RELATION_JSON = {
            'train': f'{self.AIC22_META_ROOT}/relation/train_tracks_relation.json',
            'test': f'{self.AIC22_META_ROOT}/relation/test_tracks_relation.json',
        }

        ### Generated from `scripts/action/stop_turn_det.py`
        self.STOP_TURN_JSON = {
            'train': f'{self.AIC22_META_ROOT}/action/train_stop_turn.json',
            'test': f'{self.AIC22_META_ROOT}/action/test_stop_turn.json',
        }

        ### For test tracks only (because these dont have query text), generated from `tools/predict_srl.sh`
        self.VEHICLE_JSON = {
            'train': f'{self.AIC22_ROOT}/results/classification/train_tracks/vehicle_prediction.json',
            'test': f'{self.AIC22_ROOT}/results/classification/test_tracks/vehicle_prediction.json',
        }
        self.COLOR_JSON = {
            'train': f'{self.AIC22_ROOT}/results/classification/train_tracks/color_prediction.json',
            'test': f'{self.AIC22_ROOT}/results/classification/test_tracks/color_prediction.json',
        }
        self.COLOR_MAPPING = {0: 'silver', 1: 'orange', 2: 'red', 3: 'blue', 4: 'black', 5: 'purple', 6: 'green', 7: 'white'}
        self.VEHICLE_MAPPING = {0: 'prius', 1: 'wagon', 2: 'van', 3: 'jeep', 4: 'pickup', 5: 'trailer'}

        ## Camera ids
        self.TEST_CAM_IDS = [
                'S01/c001', 'S01/c002', 'S01/c003', 'S01/c004', 'S01/c005', 
                'S03/c010', 'S03/c011', 'S03/c012', 'S03/c013', 'S03/c014', 'S03/c015', 
                'S04/c016', 'S04/c017', 'S04/c018', 'S04/c019', 'S04/c020', 'S04/c021', 'S04/c022', 'S04/c023',
                'S04/c024', 'S04/c025', 'S04/c026', 'S04/c027', 'S04/c028', 'S04/c029', 'S04/c030', 'S04/c031',
                'S04/c032', 'S04/c033', 'S04/c034', 'S04/c035', 'S04/c036', 'S04/c037', 'S04/c038', 'S04/c039', 'S04/c040'
        ]

        self.TRAIN_CAM_IDS = [
            'S02/c006', 'S02/c007', 'S02/c008', 'S02/c009', 
            'S05/c010', 'S05/c016',  'S05/c017',  'S05/c018', 'S05/c019', 'S05/c020', 
            'S05/c021', 'S05/c022',  'S05/c023', 'S05/c024', 'S05/c025',  'S05/c026', 
            'S05/c027', 'S05/c028',  'S05/c029', 'S05/c033', 'S05/c034',  'S05/c035', 'S05/c036'
        ]