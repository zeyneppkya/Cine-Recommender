# Akıllı Film Öneri Sistemi

## Genel Bakış
Üç farklı algoritma kullanarak akıllı film önerileri sunan Flask tabanlı bir web uygulaması:
- **İçerik Tabanlı Filtreleme**: Film türlerinde TF-IDF ve cosine similarity kullanır
- **İşbirlikçi Filtreleme**: Kullanıcı puanlama kalıplarını kullanır
- **Hibrit Sistem**: Daha iyi öneriler için her iki yaklaşımı birleştirir

## Proje Yapısı
```
├── app.py                 # Ana Flask uygulaması ve öneri motorları
├── movies.csv             # Film verileri (movieId, title, genres)
├── ratings.csv            # Kullanıcı puanları (userId, movieId, rating)
├── templates/
│   └── index.html         # Ana web arayüzü
├── static/
│   └── style.css          # Stil dosyası
└── replit.md              # Bu dokümantasyon dosyası
```

## Nasıl Çalıştırılır
Run butonuna tıklayın veya `python app.py` komutunu çalıştırın. Uygulama 5000 portunda başlayacaktır.

## API Endpoints
- `GET /` - Ana web arayüzü
- `POST /api/recommend` - Bir film için öneriler al
- `GET /api/movies` - Tüm filmlerin listesini al

## Öneri Algoritmaları

### İçerik Tabanlı Filtreleme
Film türlerini vektörize etmek için TF-IDF (Term Frequency-Inverse Document Frequency) kullanır ve filmler arasındaki cosine similarity'yi hesaplar.

### İşbirlikçi Filtreleme
Kullanıcı-film puanlama matrisi oluşturur ve kullanıcıların filmleri nasıl puanladığına göre film-film benzerliği hesaplar.

### Hibrit Sistem
Her iki yöntemi ağırlıklı ortalama ile birleştirir (varsayılan %50 her biri) daha güçlü öneriler için.

## Açıklanabilir Öneriler
Her öneri için kullanıcıya neden bu filmin önerildiği açıklanır:
- "Bu filmi aksiyon filmlerini sevdiğiniz için öneriyoruz"
- "Benzer kullanıcılar bu filmi yüksek puanladı"
- "Bu film hem tür benzerliği hem de kullanıcı tercihleri açısından size uygun"

## Bağımlılıklar
- Flask
- pandas
- scikit-learn
- numpy

## Son Değişiklikler
- Aralık 2025: Türkçe arayüz ve Netflix temalı tasarım
- Aralık 2025: Açıklanabilir öneriler (Explainability) özelliği eklendi
- Aralık 2025: "How It Works" bölümü kaldırıldı, daha sade arayüz
