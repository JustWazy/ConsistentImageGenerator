# Stable Diffusion XL Optimized Pipeline 🚀

Repository ini berisi built-in script teroptimasi untuk **Stable Diffusion XL 1.0** dengan manajemen memori, konsistensi karakter, image-to-image reference, dan kontrol dinamis.

---

## Fitur Utama ✨
- ✅ **Optimasi VRAM** (12-13GB) dengan CPU offloading & mixed precision
- ✅ **Dual Pipeline**: Base Model + Refiner untuk detail maksimal
- ✅ **Image-to-Image** dengan referensi gambar sebelumnya
- ✅ Kontrol resolusi dinamis (768x768 hingga 1152x1152)
- ✅ Interactive CLI untuk iterasi cepat
- ✅ Support semua jenis generasi: Realis, Anime, Karakter, Landscape

---
## Benchmark Google Colab (T4 GPU) 🏎
|Resolusi    |Langkah |VRAM Usage  | Waktu   |
|------------|--------|------------|---------|
| 896x896    | 30+15  | ~12.5GB    | 45-55s  |
| 1024x1024  | 25+10  | ~13.1GB    | 35-45s  |
