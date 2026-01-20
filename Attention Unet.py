class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class UpAtt(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        self.att = AttentionGate(F_g=in_ch//2, F_l=skip_ch, F_int=out_ch)
        self.conv = DoubleConv(in_ch//2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x2 = self.att(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUNetWithClassifier(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*16)

        self.up1 = UpAtt(base*16, base*8, base*8)
        self.up2 = UpAtt(base*8,  base*4, base*4)
        self.up3 = UpAtt(base*4,  base*2, base*2)
        self.up4 = UpAtt(base*2,  base,   base)
        self.outc = nn.Conv2d(base, 1, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*16, base*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base*8, num_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.down4(x4)

        cls_logits = self.cls_head(self.gap(xb))

        x = self.up1(xb, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        seg_logits = self.outc(x)
        return seg_logits, cls_logits

att_unet = AttentionUNetWithClassifier(in_ch=1, num_classes=NUM_CLASSES, base=32)
hist_att = train_joint(att_unet, train_loader, val_loader, epochs=10, lr=1e-3, lambda_cls=0.5)

plot_hist(hist_att, prefix="Attention U-Net ")

print("U-Net best val IoU:", max(hist_unet["val_iou"]))
print("Att best val IoU:", max(hist_att["val_iou"]))
print("U-Net best val Acc:", max(hist_unet["val_acc"]))
print("Att best val Acc:", max(hist_att["val_acc"]))

show_random_result(att_unet, val_ds)
show_random_result(att_unet, val_ds)

for _ in range(10):
    show_random_result(att_unet, val_ds)
