import os
import sys
from tqdm import tqdm
import cv2
from imitator.utils.util_pyrenderer import Facerender

class render_helper():
    def __init__(self, config={}):
        if len(config) == 0:
            config["flame_model_path"] = os.path.join("FLAMEModel/model/generic_model.pkl")
            config["batch_size"] = 1
            config["shape_params"] = 0
            config["expression_params"] = 100
            config["pose_params"] = 0
            config["number_worker"] = 8
            config["use_3D_translation"] = False

        from FLAMEModel.FLAME import FLAME
        self.face_model = FLAME(config)
        self.image_size = (512, 512)
        self.face_render = Facerender()

    def visualize_meshes(self, out_dir, out_seq_name, pred_vertices, audio_file=None):

        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            out_pred_rendered_images.append(pred_frame)

        # create the video out file
        out_vid_file = os.path.join(out_dir, out_seq_name+".mp4")
        video_file = self.compose_write_video(out_vid_file, out_pred_rendered_images, self.image_size)

        if audio_file is not None:
            out_vid_w_audio_file = os.path.join(out_dir, out_seq_name + "_wAudio.mp4")
            self.add_audio_to_video(audio_file, video_file, out_vid_w_audio_file)
            out_vid_file = out_vid_w_audio_file

        return out_vid_file, out_pred_rendered_images

    def render_exprs_to_image(self, exprs, shape_params=None):
        """
        exprs:  N x 103 (jawpose, exprs)
        """
        if exprs.shape[0] < 200:
            jaw_pose = exprs[:3].view(1, -1)
            exprs_only = exprs[3:].view(1, -1)
            flame_vertices = self.face_model.morph(expression_params=exprs_only, jaw_pose=jaw_pose, shape_params=shape_params)[0]
        else:
            flame_vertices = exprs.reshape(-1, 3)

        rendered_frame = self.render_images(flame_vertices)
        return rendered_frame

    def render_images(self, vertices):
        self.face_render.add_face(vertices.cpu().numpy(), self.face_model.faces)
        colour = self.face_render.render()
        return colour

    def compose_write_video(self, out_vid_file, gt_frames,
                            frame_size=(512, 512), fps=30):

        print('Writing video', out_vid_file)
        print()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(out_vid_file, fourcc, fps, frame_size)
        for i, frame in tqdm(enumerate(gt_frames), desc="writing video"):
            out_frame = frame
            writer.write(out_frame)
        writer.release()

        return out_vid_file

    def add_audio_to_video(self, audio_file, video_file, out_vid_w_audio_file):

        print("Audio file", audio_file)
        ffmpeg_command = f"ffmpeg -y -i {video_file} -i {audio_file} -map 0:v -map 1:a -c:v copy -shortest {out_vid_w_audio_file}"
        os.system(ffmpeg_command)
        print("added audio to the video", out_vid_w_audio_file)

        if sys.platform.startswith('win'):
            rm_command = ('del "{0}" '.format(video_file))
            print(rm_command)
        else:
            rm_command = ('rm {0} '.format(video_file))
        
        os.system(rm_command)
        print("removed ", video_file)

