#U NET
unet = UNetWithClassifier(in_ch=1, num_classes=NUM_CLASSES, base=32)
hist_unet = train_joint(unet, train_loader, val_loader, epochs=10, lr=1e-3, lambda_cls=0.5)

plot_hist(hist_unet, prefix="U-Net ")

@torch.no_grad()
def show_random_result(model, dataset):
    model.eval()
    idx = random.randint(0, len(dataset)-1)
    img_t, mk_t, y_t, im_path, mk_path = dataset[idx]

    seg_logits, cls_logits = model(img_t.unsqueeze(0).to(DEVICE))
    pred = (torch.sigmoid(seg_logits).cpu().numpy()[0,0] > 0.5).astype(np.float32)

    img = img_t.squeeze(0).numpy()
    gt  = mk_t.squeeze(0).numpy()

    iou = iou_score_from_logits(seg_logits.cpu(), mk_t.unsqueeze(0))
    acc = cls_accuracy(cls_logits.cpu(), y_t.unsqueeze(0))

    show_result_grid(img, gt, pred, processed=None, cls_acc=acc, iou=iou)

    print("Image:", im_path)
    if len(class_dirs)>0:
        print("Pred class id:", cls_logits.argmax(dim=1).item(), "| True id:", y_t.item())

show_random_result(unet, val_ds)
show_random_result(unet, val_ds)

for _ in range(10):
    show_random_result(unet, val_ds)
