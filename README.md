# Akıllı Film Öneri Sistemi

## Proje Özeti
Üç farklı algoritma kullanarak akıllı film önerileri sunan Flask tabanlı bir web uygulaması:
- **İçerik Tabanlı Filtreleme**: Film türlerinde TF-IDF ve kosinüs benzerliği kullanır
- **İşbirlikçi Filtreleme**: Kullanıcı puanlama kalıplarını kullanır
- **Hibrit Sistem**: Daha iyi öneriler için her iki yaklaşımı birleştirir

## Proje Yapısı
```
├── app.py                 # Ana Flask uygulaması ve öneri motorları
├── movies.csv             # Film verileri (movieId, title, genres)
├── ratings.csv            # Kullanıcı puanları (userId, movieId, rating)
├── pyproject.toml         # Proje bağımlılıkları ve meta verileri
├── uv.lock                # Tekrarlanabilir derlemeler için bağımlılık kilidi
├── templates/
│   └── index.html         # Ana web arayüzü
├── static/
│   └── style.css          # Stil dosyası
└── README.md              # Bu dokümantasyon dosyası
```

## Yerel Ortamda Çalıştırma
1. Python 3.11 veya üstü sürümün kurulu olduğundan emin olun
2. Bağımlılıkları uv kullanarak yükleyin:
   ```bash
   uv sync
   ```
   Veya pip kullanarak:
   ```bash
   pip install -e .
   ```
3. Uygulamayı çalıştırın:
   ```bash
   python app.py
   ```
4. Uygulamaya `http://127.0.0.1:5000` adresinden erişin

## Ortam Değişkenleri
Production dağıtımı için aşağıdaki ortam değişkenini ayarlayın:
- `SESSION_SECRET` - Flask oturumları için güvenli bir gizli anahtar

Linux/macOS örneği:
```bash
export SESSION_SECRET="your-secure-secret-key"
```

Windows örneği:
```cmd
set SESSION_SECRET=your-secure-secret-key
```

Not: SESSION_SECRET ayarlanmazsa uygulama varsayılan geliştirme anahtarını kullanacaktır. Bu yerel geliştirme için uygundur ancak production için değildir.

## Bağımlılıklar
- Flask >= 3.1.2
- pandas >= 2.3.3
- scikit-learn >= 1.8.0
- numpy >= 2.3.5

## API Uç Noktaları
- `GET /` - Ana web arayüzü
- `POST /api/recommend` - Bir film için öneriler al
- `GET /api/movies` - Tüm filmlerin listesini al

## Canlı Demo
[Canlı Demo Bağlantısı](#) *(Yer tutucu - gerçek dağıtım URL'si ile güncellenecek)*

## Öneri Algoritmaları

### İçerik Tabanlı Filtreleme
Film türlerini vektörize etmek için TF-IDF (Terim Frekansı-Ters Belge Frekansı) kullanır ve filmler arasındaki kosinüs benzerliğini hesaplar.

### İşbirlikçi Filtreleme
Kullanıcı-film puanlama matrisi oluşturur ve kullanıcıların filmleri nasıl puanladığına göre film-film benzerliği hesaplar.

### Hibrit Sistem
Daha güçlü öneriler için her iki yöntemi ağırlıklı ortalama ile birleştirir (varsayılan %50 her biri).

## Açıklanabilir Öneriler
Her öneri için kullanıcıya neden bu filmin önerildiği açıklanır:
- "Bu filmi aksiyon filmlerini sevdiğiniz için öneriyoruz"
- "Benzer kullanıcılar bu filmi yüksek puanladı"
- "Bu film hem tür benzerliği hem de kullanıcı tercihleri açısından size uygun"