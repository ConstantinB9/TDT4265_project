import configs.ssd300 as cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patch

def main():
    img_size = cfg.train["imshape"]
    
    feature_map_idx = 3
    
    feature_size = cfg.anchors["feature_sizes"][feature_map_idx]
    stride = cfg.anchors["strides"][feature_map_idx]
    aspect_ratio = cfg.anchors["aspect_ratios"][feature_map_idx]
    min_size = cfg.anchors["min_sizes"][feature_map_idx]
    next_min_size = cfg.anchors["min_sizes"][feature_map_idx + 1]
    
    fig, ax = plt.subplots()
    
    for i in range(5):
        for j in range(5):
            x = stride[0]*(i+0.5)
            y = stride[1]*(j+0.5)
            ax.plot([x], [y], 'bo')
            print(f"({x}, {y})")
            
            box_dx = [
                min_size[0],
                (min_size[0]*next_min_size[0])**0.5,
                *[min_size[0] * ratio**0.5 for ratio in aspect_ratio],
                *[min_size[0] / ratio**0.5 for ratio in aspect_ratio]
            ]
            box_dy = [
                min_size[1],
                (min_size[1]*next_min_size[1])**0.5,
                *[min_size[1] / ratio**0.5 for ratio in aspect_ratio],
                *[min_size[1] * ratio**0.5 for ratio in aspect_ratio]
            ]
            for dx, dy in zip(box_dx, box_dy):
                ax.add_patch(
                    patch.Rectangle(
                        (x - dx/2, y - dy/2),
                        dx,
                        dy,
                        edgecolor="black",
                        facecolor="none"
                    )
                )
    plt.savefig('task4d.png')
    plt.show()
    
    
if __name__ == "__main__":
    main()