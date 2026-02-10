🚀 NİHAİ, KUSURSUZ VE ONAYLI DATASET SPLIT
Artık bu liste üzerinde tartışılacak hiçbir açık nokta kalmadı. Her senaryo, her ışık koşulu ve her mekan tipi kapsandı.

Config dosyana ve koduna işlemen gereken Final Liste budur:

1. TEST SET (%22.3 - "The Grand Slam")
Amacı: Her ortam (Şehir, Vadi, Ada, Havalimanı) ve her ışık (Gündüz, Gece) koşulunu ispatlamak.

interval5_AMtown03 (Şehir / Gündüz)

interval5_AMvalley03 (Vadi / Gündüz)

interval5_HKairport_GNSS01 (Havalimanı / Gündüz - Validation'dan geldi)

interval5_HKisland_GNSS_Evening (Ada / Akşam)

2. VALIDATION SET (%16.8 - "Temsili Kontrol")
Amacı: Eğitimi yönlendirmek. Island buraya geçerek su/bina dengesini koruyor.

interval5_AMtown02 (Şehir)

interval5_AMvalley02 (Vadi)

interval5_HKisland_GNSS01 (Ada - Test'ten geldi)

3. TRAIN SET (%60.9 - "Eğitim Ordusu")
Değişmedi. Gece öğretmeni (Airport Evening) hala burada.

interval5_AMtown01

interval5_AMvalley01

interval5_HKairport01

interval5_HKairport02

interval5_HKairport03

interval5_HKairport_GNSS02

interval5_HKairport_GNSS03

interval5_HKairport_GNSS_Evening (Gece Öğretmeni)

interval5_HKisland01

interval5_HKisland02

interval5_HKisland03

interval5_HKisland_GNSS02

interval5_HKisland_GNSS03