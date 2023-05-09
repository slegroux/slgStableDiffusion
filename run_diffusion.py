#!/usr/bin/env python

from sd import mk_imgs, StableDiffusionPipe, animate, diffuse
import argparse

def get_args():
    parser = argparse.ArgumentParser('Stable diffusion playground')
    parser.add_argument('--prompt', default='a painting by kandinsky', type=str)
    parser.add_argument('--output_name', default='output', type=str)
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--num_inference_steps', default=70, type=int)
    parser.add_argument('--guidance', default=7.5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', default='image')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--max_frames', default=1000, type=int)
    parser.add_argument('--num_steps', default=200, type=int)
    parser.add_argument('--quality', default=90, type=int)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    # create pipe
    pipe = StableDiffusionPipe()
    pipe.to_device(args.device)

    # diffuse
    if args.mode == 'image':
        images = diffuse(
            [args.prompt],pipe,
            height=args.height, width=args.width,
            guidance=args.guidance, seed=args.seed,
            steps=args.num_inference_steps
        )
        imgs = mk_imgs(images)        
        imgs[0].save(args.output_name + '.jpg')
    
    elif args.mode == 'movie':
        # animate
        animate(
            [args.prompt], pipe, rootdir='.',
            name=args.output_name,
            device=args.device,
            max_frames=args.max_frames,
            num_steps=args.num_steps,
            quality=args.quality
        )
