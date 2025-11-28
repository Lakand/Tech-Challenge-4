# Arquivo: test_full_system.py
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def run_test():
    print("🚀 INICIANDO TESTE COMPLETO DO SISTEMA (GPU + MLFLOW)...")
    
    # ---------------------------------------------------------
    # 1. TESTE DE TREINAMENTO
    # ---------------------------------------------------------
    print("\n[1/2] Solicitando Treinamento (/train)...")
    print("      (Isso pode demorar um pouco dependendo da GPU/CPU...)")
    
    train_payload = {
        "symbol": "DIS",
        "start_date": "2020-01-01", # Periodo menor para ser rápido
        "end_date": "2023-01-01",
        "epochs": 3,                # Poucas épocas para teste rápido
        "batch_size": 32
    }
    
    try:
        response = requests.post(f"{BASE_URL}/train", json=train_payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Treinamento Concluído!")
            print(f"   MLflow Run ID: {data.get('mlflow_run_id')}")
            print(f"   Mensagem: {data.get('message')}")
        else:
            print(f"❌ Erro no Treinamento: {response.text}")
            return # Para o teste se falhar aqui
            
    except Exception as e:
        print(f"❌ Erro de conexão com a API: {e}")
        print("   Verifique se o 'python app.py' está rodando em outro terminal.")
        return

    # ---------------------------------------------------------
    # 2. TESTE DE PREVISÃO (INFERÊNCIA)
    # ---------------------------------------------------------
    print("\n[2/2] Solicitando Previsão (/predict)...")
    
    predict_payload = {
        "symbol": "DIS",
        "lookback_days": 60
    }
    
    start_req = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=predict_payload)
    total_time = time.time() - start_req
    
    if response.status_code == 200:
        data = response.json()
        perf = data.get("performance", {})
        
        print(f"✅ Previsão Recebida!")
        print(f"   Preço Previsto: $ {data.get('predicted_close_price'):.2f}")
        print(f"   ------------------------------------------------")
        print(f"   📊 MONITORAMENTO DE PERFORMANCE (RETORNADO PELA API):")
        print(f"   ⏱️  Latência Interna (API): {perf.get('latency_sec')} s")
        print(f"   ⏱️  Latência Total (Request): {total_time:.4f} s")
        print(f"   💾 Uso de RAM: {perf.get('ram_usage_mb')} MB")
        
        # Verifica se GPU foi usada
        if "gpu_vram_mb" in perf:
            print(f"   🎮 GPU VRAM Usada: {perf.get('gpu_vram_mb')} MB")
            print(f"   🔥 GPU Utilização: {perf.get('gpu_util_percent')} %")
        else:
            print("   ⚠️  GPU não detectada ou monitoramento desativado.")
            
    else:
        print(f"❌ Erro na Previsão: {response.text}")

if __name__ == "__main__":
    run_test()