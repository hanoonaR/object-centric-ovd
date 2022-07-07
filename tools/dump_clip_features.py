import argparse
import json
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/coco/annotations/instances_val2017.json')
    parser.add_argument('--out_path', default='zeroshot_weights/coco_clip_a+photo+cname.npy')
    parser.add_argument('--prompt', default='photo')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="ViT-B/32")
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in \
                 sorted(data['categories'], key=lambda x: x['id'])]
    if 'synonyms' in data['categories'][0]:
        synonyms = [x['synonyms'] for x in \
                    sorted(data['categories'], key=lambda x: x['id'])]
    else:
        synonyms = []
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using prompt, ", args.prompt)

    if args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
                              for x in synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
                              for x in synonyms]

    print('sentences_synonyms', len(sentences_synonyms), \
          sum(len(x) for x in sentences_synonyms))
    if args.model == 'clip':
        import clip

        print('Loading CLIP')
        model, preprocess = clip.load(args.clip_model, device=device)
        text = clip.tokenize(sentences).to(device)
        with torch.no_grad():
            if len(text) > 10000:
                text_features = torch.cat([
                    model.encode_text(text[:len(text) // 2]),
                    model.encode_text(text[len(text) // 2:])],
                    dim=0)
            else:
                text_features = model.encode_text(text)
        print('text_features.shape', text_features.shape)
        text_features = text_features.cpu().numpy()
    else:
        assert 0, args.model
    if args.out_path != '':
        print('saveing to', args.out_path)
        np.save(open(args.out_path, 'wb'), text_features)
