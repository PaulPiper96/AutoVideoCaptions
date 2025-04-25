from scenedetect import detect, ContentDetector
import imageio
import matplotlib.pyplot as plt
import os



def detectcuts(path):
    scene_list = detect(path, ContentDetector())
    framearray = []
    for scene in scene_list:
        end_frame = scene[1].get_frames()
        print('Scene End Frame %d' % end_frame)
        framearray.append(end_frame)
    return framearray


def save_images_with_matplotlib(frame_list, indices, save_dir="frames", prefix=""):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for i in indices:
        if i < len(frame_list):
            frame = frame_list[i]
            plt.imshow(frame)
            plt.axis('off')

            # Build full file path
            filename = os.path.join(save_dir, f"{prefix}_{i}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

            print(f"Saved frame {i} to {filename}")
        else:
            print(f"Index {i} out of range, skipped.")

def show_image(frame, title="Frame"):
    plt.imshow(frame)
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Path to your video
    videopath_ = r"C:\Users\paulp\Desktop\autocaptionsflh\testvideos\Movie30S.mp4"

    # Step 1: Detect scene cuts
    scene_end_frames = detectcuts(videopath_)

    # Step 2: Load first N frames into a list (you can increase this)
    reader = imageio.get_reader(videopath_)
    frame_list = [frame for i, frame in enumerate(reader) if i < 5000]
    reader.close()

    max_index = len(frame_list) - 1
    
    saveimagestodisk=[]

    # Step 3: Show cut frame and next 3 frames
    for cut_frame in scene_end_frames:
        idx = cut_frame - 1  # Convert 1-based frame number to 0-based index

        if idx <= max_index:
            #show_image(frame_list[idx], title=f"Scene Cut at Frame {cut_frame}")

            # Show next 3 frames if available
         
            next_idx = idx +1
            if next_idx <= max_index:
                #show_image(frame_list[next_idx], title=f"Frame {cut_frame + 1} (After Cut)")
                saveimagestodisk.append(idx-20)

    
    save_images_with_matplotlib(frame_list, saveimagestodisk)