import torch
import numpy as np
import os, datetime, glob
from timeit import default_timer as timer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import librosa
import pickle
from omegaconf import OmegaConf
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from imitator.test.test_model_voca import get_latest_checkpoint
from imitator.utils.render_helper import render_helper
from imitator.utils.init_from_config import instantiate_from_config


class test_on_audio():

    def __init__(self):
        if os.getenv("WAV2VEC_PATH"):
            wav2vec_path = os.getenv("WAV2VEC_PATH")
        else:
            wav2vec_model = "projects/dataset/voca_face_former/wav2vec2-base-960h"
            wav2vec_path = os.path.join(os.getenv('HOME'), wav2vec_model)
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        self.rh = render_helper()

        if os.getenv("VOCASET_PATH"):
            template_file = os.path.join(os.getenv("VOCASET_PATH"), "templates.pkl")
        else:
            template_file = os.path.join(os.getenv("HOME"), "projects/dataset/voca_face_former", "templates.pkl")
        with open(template_file, 'rb') as handle:
            self.templates = pickle.load(handle, encoding='latin1')

    def read_audio_from_file(self, wav_path):
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)
        return input_values

    def load_model_from_checkpoint(self, model_ckpt):
        def find_index_from_list_with_partial_match(path_list, search_Str):
            for i, sub_string in enumerate(path_list):
                if search_Str in sub_string:
                    break
            return i

        paths = model_ckpt.split("/")
        idx = len(paths) - find_index_from_list_with_partial_match(paths[::-1], "log") + 1
        logdir = "/".join(paths[:idx])
        print("Logdir", logdir)
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        print("base_configs", base_configs)
        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        configs = OmegaConf.merge(*configs)

        model = instantiate_from_config(configs.model)
        model.init_from_ckpt(model_ckpt)  # init the model
        # freeze the params
        return model

    def run_on_wav_file(self, template, model,
                            audio_file, condition_for_tesing, subj_name,
                            out_dir):
        """
        # Load the model to be used for testing
        # Load the audio files in the path -> Done
        # run the audio inferencer -> Done
        # load the template file and generate one hot encoding
        # run the model.imitator
        # render them as vidoe and add audio to the video

        template :
        """
        # audio file
        processed_audio = self.read_audio_from_file(audio_file)
        sampled_processed_audio = torch.from_numpy(processed_audio).view(1, -1)
        sampled_processed_audio = sampled_processed_audio

        # run the prediction and out formate : Bs x Nf X 15059
        template = torch.from_numpy(template).view(1, -1)
        file_name = audio_file.split("/")[-1]

        print()
        result_npy_dict = {}
        for condition in condition_for_tesing:
            seq_name_w_condition = subj_name + "_" + file_name.split(".wav")[0] + "_condition_" + str(condition)
            print("seq name with condition", seq_name_w_condition)

            # generate the one hot encoding for the model
            one_hot = torch.zeros((1, 8))
            one_hot[0, condition] = 1

            # ************************************************************************************************************
            # use the faceformer normal prediction
            prediction = model.nn_model.predict(sampled_processed_audio, template, one_hot)
            prediction = prediction.reshape(prediction.shape[1], -1, 3).detach()

            # create the out dir and out seq name
            self.rh.visualize_meshes(out_dir, seq_name_w_condition, prediction, audio_file)
            result_npy_dict[seq_name_w_condition] = prediction.detach().cpu().numpy()

        return result_npy_dict

    def run_test_on_the_audio_path(self, exp_cfg, model_folder, model_ckpt):

        model = self.load_model_from_checkpoint(model_ckpt)
        model.eval()
        if opt.out_dir != "None":
            out_dir = os.path.join(model_folder, "external_audio_results")
        else:
            out_dir = opt.out_dir
        os.makedirs(out_dir, exist_ok=True)

        ### set up the testing
        identity_for_testing = exp_cfg.test_subject.split(" ")
        condition_for_tesing = [int(x) for x in exp_cfg.test_condition.split(" ")]
        print("identity_for_testing", identity_for_testing)
        print("condition_for_tesing", condition_for_tesing)

        all_results = {}
        for subj_name in identity_for_testing:
            template = self.templates[subj_name]
            file_name = opt.audio.split("/")[-1]
            seq_name = subj_name + "_" + file_name.split(".wav")[0]
            out_results = self.run_on_wav_file(template,
                                               model,
                                               opt.audio,
                                               condition_for_tesing,
                                               subj_name,
                                                out_dir)
            all_results[seq_name] = out_results

        # store the results on the model
        if exp_cfg.dump_results:
            out_file = os.path.join(out_dir, "results_dict.npy")
            np.save(out_file, all_results)
            print("Dumping out file", out_file)

def get_parser(parser):

    all_voca_subjects = [
                        ## train
                        "FaceTalk_170728_03272_TA",
                         "FaceTalk_170904_00128_TA,"
                         "FaceTalk_170725_00137_TA",
                         "FaceTalk_170915_00223_TA",
                         "FaceTalk_170811_03274_TA",
                         "FaceTalk_170913_03279_TA",
                         "FaceTalk_170904_03276_TA",
                         "FaceTalk_170912_03278_TA",
                         ## val subjects
                         "FaceTalk_170811_03275_TA",
                         "FaceTalk_170908_03277_TA",
                        ## test subjects
                        "FaceTalk_170809_00138_TA",
                        "FaceTalk_170731_00024_TA"
                        ]
    parser.add_argument('-g', "--gpus", type=str, default=0)
    parser.add_argument('-m', "--model", type=str, default="pretrained_model/generalized_model_mbp_vel")
    parser.add_argument('-o', "--out_dir", type=str, default=None)
    parser.add_argument('-a', "--audio", type=str, required=True)
    parser.add_argument('-t', '--test_subject', type=str, default="FaceTalk_170809_00138_TA",
                        choices=all_voca_subjects)
    parser.add_argument('-c', '--test_condition', type=str, default="2")
    parser.add_argument('-d', '--dump_results', action='store_true')
    parser.add_argument('-r', '--render_results', action='store_true')
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.set_defaults(unseen=False)
    return parser

if __name__ == "__main__":

    start = timer()

    parser = ArgumentParser()
    parser = get_parser(parser)
    opt = parser.parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(opt.seed)
    tester = test_on_audio()

    # run on the best model
    best_ckpt = get_latest_checkpoint(os.path.join(opt.model, "checkpoints"))
    tester.run_test_on_the_audio_path(opt, opt.model, best_ckpt)
    end = timer()
    print("\n\nTime to take to run the testing suite in sec", end - start)