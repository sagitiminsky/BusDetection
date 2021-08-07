from train import get_efficientdet_config, create_model, create_loader
# import yaml
from data.bus_dataset import BusDataSet
from data.transforms import *
import os

print('when running')
##

def run(myAnnFileName='test/annotationsPred.txt', buses='test/busesTest',
        model_best='model_best/model_best.pth.tar'):
    print('inside runMe!')
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        os.remove(myAnnFileName)
    except FileNotFoundError:
        pass

    print('Remove previous annotation file - success!')

    image_list = []
    for file_name in os.listdir(buses):
        file_path = os.path.join(buses, file_name)
        image = Image.open(file_path)
        image_list.append(np.asarray(image))

    print('create image list - done!')

    config = get_efficientdet_config()  # args.model
    n_class = 6
    config.num_classes = n_class  # TODO:get from dataset

    print('trying to create model ...')

    model = create_model(
        config,
        bench_task='predict',
        checkpoint_path=model_best
    )
    model = model.to(working_device)

    print("model create - done!")

    loader = create_loader(
        BusDataSet(image_list, [[np.zeros(4)]] * len(image_list), [[np.zeros(1)]] * len(image_list)),
        input_size=(3, 640, 640),
        batch_size=1
    )
    print("model loader - done!")

    output = predict(model, loader, file_names=os.listdir(buses), working_device=working_device)

    with open(myAnnFileName, 'a') as the_file:
        for image_name, bboxes in output.items():
            the_file.write(
                "{}:{}\n".format(image_name, str([[int(value) for value in box] for box in bboxes])[1:-1]))

    print("Done runMe !")


# https://github.com/rwightman/efficientdet-pytorch/issues/88
def predict(model, loader, file_names, score_threshold=0.22, working_device='cpu'):
    model.eval()

    response = {}

    with torch.no_grad():
        for tup, file_name in zip(loader, file_names):
            #  BATCH SIZE WAS SET TO 1

            image = tup[0]  # loader->(input,target)
            output = model(image.to(device=working_device),
                           img_scales=5.7 * torch.ones(image.shape[0]).to(device=working_device),
                           img_size=(torch.tensor([[3648, 2736]])).to(device=working_device))

            boxes = output.detach().cpu().numpy()[0][:, :4]  # bb
            scores = output.detach().cpu().numpy()[0][:, 4]  # prob
            cls = output.detach().cpu().numpy()[0][:, 5]  # class
            mask = scores > score_threshold
            response[file_name] = combine_box_index(masked_array(boxes, mask), masked_array(cls, mask))

    return response


def combine_box_index(boxes, classes):
    output = []
    for box, _class in zip(list(boxes), list(classes)):
        output.append(np.append(box, _class))

    return output


def masked_array(a, mask):
    output = []
    for i, b in enumerate(mask):
        if b:
            output.append(a[i])
    return output


if '__main__' == __name__:
    run()
