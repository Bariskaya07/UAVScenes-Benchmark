# Multi-Modal Segmentation - Kanal Yönetimi Özeti

**Tarih:** Şubat 2025
**Proje:** UAVScenes + DELIVER Multi-Modal Semantic Segmentation Benchmark

---

## HIZLI ÖZET (TL;DR)

| Soru | Cevap |
|------|-------|
| UAVScenes çalışıyor mu? | **EVET** - 3-ch stack ile eskisi gibi |
| DELIVER çalışacak mı? | **EVET** - Aynı yaklaşım (1-ch → 3-ch stack) |
| Model değişikliği gerekli mi? | **HAYIR** - Tüm modeller 3 kanal bekliyor |
| Şimdi ne yapmam lazım? | **HİÇBİR ŞEY** - Deneylerini çalıştır |

---

## 1. TEMEL PRENSİP

### Orijinal CMNeXt Yaklaşımı (CVPR 2023)

Orijinal CMNeXt kodunu inceledik (`https://github.com/InSAI-Lab/DELIVER`):

**Model:** Tüm modaliteler 3 kanal bekliyor
```python
PatchEmbedParallel(3, embed_dims[0], 7, 4, 7//2, self.num_modals)
#                  ^ 3 kanal sabit
```

**Dataset:** 1 kanallı veriler 3 kanala stack ediliyor
```python
def _open_img(self, file):
    img = io.read_image(file)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)  # 1-ch → 3-ch
    return img
```

### Sonuç

| Dataset | Orijinal Format | Model Girişi |
|---------|-----------------|--------------|
| UAVScenes | 1-ch HAG | 3-ch stack |
| DELIVER | 1-ch Range | 3-ch stack |

**Her iki dataset de aynı yaklaşımı kullanıyor → Model değişikliği gerekmez!**

---

## 2. MEVCUT DURUM

### Tüm Projeler için Config Değerleri

| Proje | Dataset Param | Model Param | Değer |
|-------|---------------|-------------|-------|
| CMNeXt | `AUX_CHANNELS` | `AUX_IN_CHANS` | 3 |
| DFormer | `aux_channels` | `aux_in_chans` | 3 |
| Sigma | `aux_channels` | `extra_in_chans` | 3 |
| Mul_VMamba | `AUX_CHANNELS` | `EXTRA_IN_CHANS` | 3 |
| TokenFusion | `aux_channels` | `in_chans` | 3 |
| GeminiFusion | `aux_channels` | `in_chans` | 3 |

**Tüm projeler 3 kanal kullanıyor → Backward compatible!**

---

## 3. YAPILAN DEĞİŞİKLİKLER

### 3.1 DELIVER Dizin Yapısı
```
/home/bariskaya/Projelerim/UAV/data/DELIVER/
├── img/       # RGB görüntüler (3-ch)
├── lidar/     # Range projection (1-ch → 3-ch stack)
├── hha/       # Depth HHA encoding (3-ch)
├── event/     # Event camera
└── semantic/  # Ground truth labels
```

### 3.2 UAVScenes Dataset Değişiklikleri

| Proje | Dosya | Değişiklik |
|-------|-------|------------|
| CMNeXt | `semseg/datasets/uavscenes.py` | `aux_channels=3` parametresi |
| DFormer | `utils/dataloader/UAVScenesDataset.py` | `aux_channels=3` parametresi |
| TokenFusion | `datasets/uavscenes.py` | `aux_channels=3` parametresi |
| GeminiFusion | `datasets/uavscenes.py` | `aux_channels=3` parametresi |
| Sigma | `dataloader/UAVScenesDataset.py` | `aux_channels=3` parametresi |
| Mul_VMamba | `semseg/datasets/uavscenes.py` | `aux_channels=3` parametresi |

### 3.3 DELIVER Dataset Sınıfları (Yeni)

| Proje | Dosya |
|-------|-------|
| CMNeXt | `semseg/datasets/deliver.py` |
| DFormer | `utils/dataloader/DELIVERDataset.py` |
| TokenFusion | `datasets/deliver.py` |
| GeminiFusion | `datasets/deliver.py` |
| Sigma | `dataloader/DELIVERDataset.py` |
| Mul_VMamba | `semseg/datasets/deliver.py` |

### 3.4 Model Değişiklikleri

| Proje | Dosya | Parametre |
|-------|-------|-----------|
| CMNeXt | `semseg/models/cmnext.py` | `aux_in_chans=3` |
| DFormer | `models/encoders/DFormer.py` | `aux_in_chans=3` |
| Sigma | `models/encoders/dual_segformer.py` | `extra_in_chans=3` |
| Mul_VMamba | `semseg/models/backbones/mulmamba.py` | `extra_in_chans=3` |
| TokenFusion | Shared weights mimarisi | `in_chans=3` (dataset'te) |
| GeminiFusion | Shared weights mimarisi | `in_chans=3` (dataset'te) |

### 3.5 UAVScenes Config Güncellemeleri

| Proje | Config Dosyası |
|-------|----------------|
| CMNeXt | `configs/uavscenes_rgb_hag.yaml` |
| DFormer | `local_configs/UAVScenes/DFormerv2_B.py` |
| TokenFusion | `configs/uavscenes_rgb_hag.yaml` |
| GeminiFusion | `configs/uavscenes_rgb_hag.yaml` |
| Sigma | `configs/config_UAVScenes.py` |
| Mul_VMamba | `configs/uavscenes_rgbhagmulmamba.yaml` |

### 3.6 DELIVER Config Dosyaları (Yeni)

| Proje | Config Dosyası |
|-------|----------------|
| CMNeXt | `configs/deliver_rgb_lidar.yaml` |
| DFormer | `local_configs/DELIVER/DFormerv2_B.py` |
| TokenFusion | `configs/deliver_rgb_lidar.yaml` |
| GeminiFusion | `configs/deliver_rgb_lidar.yaml` |
| Sigma | `configs/config_DELIVER.py` |
| Mul_VMamba | `configs/deliver_rgblidarmulmamba.yaml` |

---

## 4. DELIVER GEÇİŞ PLANI

DELIVER veri seti geldiğinde yapılacaklar:

### Adım 1: Veri İndirme
```bash
# DELIVER veri setini indir
# data/DELIVER/ dizinine kopyala
```

### Adım 2: Mean/Std Hesaplama
```python
import numpy as np
from glob import glob
from PIL import Image

lidar_files = glob('data/DELIVER/lidar/**/*.png', recursive=True)
values = [np.array(Image.open(f)).mean() / 255.0 for f in lidar_files[:1000]]
lidar_mean = np.mean(values)
lidar_std = np.std(values)
print(f"LiDAR: mean={lidar_mean:.4f}, std={lidar_std:.4f}")
```

### Adım 3: Config Güncelleme
```yaml
DATASET:
  AUX_MEAN: [lidar_mean, lidar_mean, lidar_mean]  # 3-ch stack
  AUX_STD: [lidar_std, lidar_std, lidar_std]
```

### Adım 4: Training
```bash
python train.py --config configs/deliver_rgb_lidar.yaml
```

---

## 5. MAKALEDE KULLANILACAK AÇIKLAMA

### İngilizce
```
Both UAVScenes and DELIVER datasets use single-channel geometric
features (HAG and Range respectively), which are stacked to 3 channels
following the standard CMNeXt protocol. This allows using ImageNet
pretrained weights without modification.
```

### Türkçe
```
Hem UAVScenes hem de DELIVER veri setleri tek kanallı geometrik
özellikler kullanmaktadır (sırasıyla HAG ve Range). Bu özellikler,
standart CMNeXt protokolüne uygun olarak 3 kanala stack edilmektedir.
Bu sayede ImageNet ön eğitimli ağırlıklar değişiklik yapılmadan
kullanılabilmektedir.
```

---

## 6. KONTROL KOMUTLARI

### Config Değerlerini Kontrol Et
```bash
# UAVScenes
grep -E "AUX_CHANNELS|AUX_IN_CHANS|aux_channels|in_chans" \
  CMNeXt_UAVScenes/configs/uavscenes_rgb_hag.yaml \
  Mul_VMamba_repo/configs/uavscenes_rgbhagmulmamba.yaml

# DELIVER
grep -E "AUX_CHANNELS|AUX_IN_CHANS|aux_channels|in_chans" \
  CMNeXt_UAVScenes/configs/deliver_rgb_lidar.yaml \
  Mul_VMamba_repo/configs/deliver_rgblidarmulmamba.yaml
```

### Dataset Kodunu Kontrol Et
```bash
grep -n "aux_channels\|stack\|repeat" */datasets/uavscenes.py
grep -n "aux_channels\|stack\|repeat" */datasets/deliver.py
```

---

## 7. SSS (Sık Sorulan Sorular)

### S: UAVScenes deneylerimi şimdi çalıştırabilir miyim?
**C: EVET.** Her şey backward compatible, eskisi gibi çalışıyor.

### S: DELIVER için ayrı bir model mi eğitmem gerekiyor?
**C: EVET.** Her dataset için ayrı model eğitilir, ama aynı mimari kullanılır.

### S: TokenFusion ve GeminiFusion neden farklı?
**C:** Bu modeller "shared weights" mimarisi kullanıyor. RGB ve aux modality için aynı encoder kullanılıyor, bu yüzden her iki modality de 3 kanala sahip olmalı.

### S: Neden 1 kanal yerine 3 kanal stack ediyoruz?
**C:** ImageNet pretrained ağırlıklarını kullanabilmek için. ImageNet 3 kanallı RGB üzerinde eğitilmiş, bu yüzden modeller 3 kanal bekliyor.

### S: 1 kanal ile çalıştırmak mümkün mü?
**C:** Evet, ama pretrained ağırlıkların ilk katmanı random init olur. Genellikle 3-ch stack daha iyi sonuç verir.

---

## 8. REFERANSLAR

- CMNeXt Paper: "Delivering Arbitrary-Modal Semantic Segmentation" (CVPR 2023)
- DELIVER Dataset: Multi-modal driving dataset with RGB, Depth, LiDAR, Event
- UAVScenes Dataset: Aerial semantic segmentation with RGB + HAG
- Orijinal CMNeXt Repo: https://github.com/InSAI-Lab/DELIVER

---

**Son Güncelleme:** Şubat 2025
