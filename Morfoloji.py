import cv2
import numpy as np

'''
# 8.jpg resmini dosya içerisinden okuma
img = cv2.imread('8.jpg', cv2.IMREAD_GRAYSCALE)

# Erosion (aşındırma) fonksiyonunu kullanarak 8a.jpg resmini oluşturma
kernel = np.ones((3,3),np.uint8)
img_a = cv2.erode(img, kernel, iterations = 1)

# Dilation (yayma) fonksiyonunu kullanarak 8b.jpg resmini oluşturma
img_b = cv2.dilate(img, kernel, iterations = 1)

# 8a ve 8b resimlerini dosyaya kaydet
cv2.imwrite('8a.jpg', img_a)
cv2.imwrite('8b.jpg', img_b)
'''

# Resimleri yükleme
img = cv2.imread('8.jpg')
img_a = cv2.imread('8a.jpg')
img_b = cv2.imread('8b.jpg')

# Boş bir pencere oluşturma
cv2.namedWindow('Morphology Operations')

# Operasyon isimleri
operation_names = ['Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat']

# Morfolojik işlemleri uygulayan fonksiyon
def apply_morphology(morph_type, kernel_size):
    # Morfolojik işlem seçenekleri
    morph_ops = {
        0: cv2.MORPH_ERODE,
        1: cv2.MORPH_DILATE,
        2: cv2.MORPH_OPEN,
        3: cv2.MORPH_CLOSE,
        4: cv2.MORPH_GRADIENT,
        5: cv2.MORPH_TOPHAT,
        6: cv2.MORPH_BLACKHAT
    }
    
    # Seçilen morfolojik işlemi uygulama
    morphed_img = cv2.morphologyEx(img, morph_ops[morph_type], np.ones((kernel_size, kernel_size), np.uint8))
    morphed_img_a = cv2.morphologyEx(img_a, morph_ops[morph_type], np.ones((kernel_size, kernel_size), np.uint8))
    morphed_img_b = cv2.morphologyEx(img_b, morph_ops[morph_type], np.ones((kernel_size, kernel_size), np.uint8))
    
    # Görsellerin isimlerini yazma
    cv2.putText(morphed_img, '8', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(morphed_img_a, '8a', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(morphed_img_b, '8b', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Sonuçları görselleştirme
    cv2.imshow('Morphology Operations', np.hstack([morphed_img, morphed_img_a, morphed_img_b]))
    cv2.waitKey(1)

# Trackbar değerlerini almak için boş bir fonksiyon tanımlama
def nothing(x):
    pass

# Trackbar'ları oluşturma
cv2.createTrackbar('Operation', 'Morphology Operations', 0, len(operation_names)-1, nothing)
cv2.createTrackbar('KernelSize', 'Morphology Operations', 1, 21, nothing)

# İlk morfolojik işlemi uygulama
apply_morphology(0, 1)

while True:
    # Trackbar değerlerini alma
    morph_type = cv2.getTrackbarPos('Operation', 'Morphology Operations')
    kernel_size = cv2.getTrackbarPos('KernelSize', 'Morphology Operations')
    
    # Morfolojik işlemi uygulama
    apply_morphology(morph_type, kernel_size)
    
    # Trackbar değerini operasyon ismine dönüştürme
    operation_name = operation_names[morph_type]
    
    # Pencere başlığına operasyon ismini ekleme
    cv2.setWindowTitle('Morphology Operations', f'Operation: {operation_name} (Kernel Size: {kernel_size})')
    
    # q tuşuna basılana kadar bekleme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()