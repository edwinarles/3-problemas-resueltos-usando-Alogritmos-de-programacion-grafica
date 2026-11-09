import os
import argparse
import glob
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import SRGANGenerator

def get_device(requested="auto"):
    if requested in ["cuda", "auto"] and torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            test_conv = torch.nn.Conv2d(1, 1, 1).cuda()
            _ = test_conv(test_tensor.unsqueeze(0).unsqueeze(0))
            return torch.device("cuda")
        except Exception:
            print("[!] GPU no compatible con esta compilacion de PyTorch. Usando CPU...")
            return torch.device("cpu")
    return torch.device("cpu")

def process_video(args):
    device = get_device(args.device)
    print(f"[*] Usando dispositivo: {device}")

    # 1. Cargar modelo Generador
    generator = SRGANGenerator(num_residual_blocks=args.res_blocks).to(device)
    if os.path.exists(args.weights):
        generator.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[+] Pesos cargados desde: {args.weights}")
    else:
        print(f"[!] Aviso: No se encontro {args.weights}. Ejecutando con pesos iniciales de prueba.")

    generator.eval()

    # 2. Localizar video de entrada
    video_path = args.video
    if not video_path or not os.path.exists(video_path):
        candidates = glob.glob("../*.mp4") + glob.glob("*.mp4")
        if candidates:
            video_path = candidates[0]
        else:
            raise FileNotFoundError("No se encontro un video de entrada (.mp4).")

    print(f"[+] Procesando video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Para no saturar memoria si el video original es grande, podemos procesar una resolucion base
    target_in_w = min(width, 320)
    target_in_h = min(height, 240)
    out_w = target_in_w * 4
    out_h = target_in_h * 4

    # Configurar VideoWriter
    os.makedirs(os.path.dirname(args.output_video) if os.path.dirname(args.output_video) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (out_w, out_h))

    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    saved_comparison = False
    frame_idx = 0

    print(f"[*] Procesando frames ({total_frames} frames)...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Redimensionar al tamano de entrada deseado
        frame_resized = cv2.resize(frame_bgr, (target_in_w, target_in_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Inferencia con SRGAN
        input_tensor = transform_to_tensor(frame_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = generator(input_tensor)

        # Desnormalizar de [-1, 1] a [0, 255]
        sr_tensor = (sr_tensor.squeeze(0).cpu().clamp(-1, 1) + 1.0) / 2.0
        sr_img_rgb = (sr_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        sr_img_bgr = cv2.cvtColor(sr_img_rgb, cv2.COLOR_RGB2BGR)

        out_writer.write(sr_img_bgr)

        # Guardar comparacion del primer tercio del video
        if not saved_comparison and frame_idx > 10:
            bicubic_bgr = cv2.resize(frame_resized, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            comp = np.hstack([bicubic_bgr, sr_img_bgr])
            # Escribir texto comparativo
            cv2.putText(comp, "Original / Bicubic 4x", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(comp, "SRGAN Mejorado 4x", (out_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite("comparacion_antes_despues.png", comp)
            saved_comparison = True

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames} procesado...")

    cap.release()
    out_writer.release()
    print(f"[OK] Video mejorado guardado en: {args.output_video}")
    if saved_comparison:
        print("[OK] Imagen de comparacion guardada en: comparacion_antes_despues.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia de SRGAN sobre un video")
    parser.add_argument("--video", type=str, default="", help="Ruta al video de entrada")
    parser.add_argument("--weights", type=str, default="checkpoints/generator_srgan.pth", help="Pesos del generador")
    parser.add_argument("--output_video", type=str, default="video_mejorado.mp4", help="Ruta del video de salida")
    parser.add_argument("--res_blocks", type=int, default=6, help="Numero de bloques residuales")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo (auto, cuda, cpu)")

    args = parser.parse_args()
    process_video(args)

