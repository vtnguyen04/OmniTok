import argparse
import datetime
import os
import subprocess
import sys
import time

LOG_FILE = "server_crash_debug.log"

def get_ram_stats():
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_total = int(lines[0].split()[1])
        mem_avail = int(lines[2].split()[1])
        return {
            "ram_total_mb": mem_total // 1024,
            "ram_avail_mb": mem_avail // 1024,
            "ram_used_mb": (mem_total - mem_avail) // 1024
        }
    except Exception:
        return {}

def get_gpu_stats():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip().split('\n')

        gpu_stats = {}
        logs = []
        for line in res:
            idx, temp, pwr, limit, mem_used, mem_tot, util = line.split(', ')
            idx = int(idx)
            gpu_stats[f"gpu_{idx}/temp"] = float(temp)
            gpu_stats[f"gpu_{idx}/power_draw"] = float(pwr)
            gpu_stats[f"gpu_{idx}/power_limit"] = float(limit)
            gpu_stats[f"gpu_{idx}/mem_used"] = float(mem_used)
            gpu_stats[f"gpu_{idx}/utilization"] = float(util)

            logs.append(f"GPU {idx}: {temp}C | Pwr {pwr}W | Mem {mem_used}MB | Util {util}%")

        return gpu_stats, " | ".join(logs)
    except Exception as e:
        return {}, f"GPU Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="WandB run ID to attach to")
    _ = parser.parse_args()

    print(f"Bắt đầu theo dõi hệ thống. Đang ghi log cục bộ vào: {os.path.abspath(LOG_FILE)}")

    # Tích hợp vào chung 1 run với train.py
    # run = wandb.init(
    #     project="omnitok",
    #     id=args.run_id,
    #     resume="allow" if args.run_id else None,
    #     name=f"SysMon_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" if not args.run_id else None,
    #     job_type="hardware_monitor",
    #     tags=["system_crash_monitor"] if not args.run_id else None,
    #     settings=wandb.Settings(init_timeout=300),
    # )

    # print(f"✅ Đã kết nối WandB: {run.url}")
    print("Vui lòng giữ script này chạy ngầm (hoặc tab tmux khác) khi train.")

    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\nNEW MONITORING SESSION: {datetime.datetime.now()}\n{'='*80}\n")
        f.flush()
        os.fsync(f.fileno())

    step = 0
    while True:
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ram_stats = get_ram_stats()
            gpu_stats, gpu_log_str = get_gpu_stats()

            # Gộp chung metrics để log lên wandb
            metrics = {"step": step}
            metrics.update(ram_stats)
            metrics.update(gpu_stats)

            # 1. LOG LÊN WANDB (Đã bị tắt)
            if metrics:
                pass
                # wandb.log(metrics, step=step)

            # 2. LOG XUỐNG DISK CỤC BỘ VÀ FSYNC NGAY LẬP TỨC
            # Lý do: WandB đẩy data qua mạng (mất ~100ms) và dùng buffer ngầm.
            # Nếu sập nguồn (chỉ mất 5ms), WandB CÓ THỂ KHÔNG KỊP GỬI gói tin cuối.
            # SSD ghi và fsync chỉ mất <1ms. Nên Disk luôn là bằng chứng sống cuối cùng.
            ram_log_str = f"RAM: {ram_stats.get('ram_avail_mb', 0)}MB Avail"
            log_line = f"[{now}] {ram_log_str} || {gpu_log_str}\n"

            with open(LOG_FILE, "a") as f:
                f.write(log_line)
                f.flush()
                os.fsync(f.fileno())

            time.sleep(1)  # Monitor với tần suất cực cao (1 giây)
            step += 1

        except KeyboardInterrupt:
            print("\nĐã dừng monitor.")
            # wandb.finish()
            sys.exit(0)
        except Exception as e:
            err_line = f"[{datetime.datetime.now()}] SCRIPT ERROR: {str(e)}\n"
            with open(LOG_FILE, "a") as f:
                f.write(err_line)
                f.flush()
                os.fsync(f.fileno())
            time.sleep(1)

if __name__ == "__main__":
    main()
