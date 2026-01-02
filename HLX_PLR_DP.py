# concurrent_run.py
import subprocess
import threading

def run_script(script_name):
    subprocess.run(["python", script_name], check=True)

if __name__ == "__main__":
    # Run both optimizations concurrently
    t1 = threading.Thread(target=run_script, args=("PLR_HLX_V3.py",))
    t2 = threading.Thread(target=run_script, args=("hlx_DP_Opt_V3.py",))

    print("Starting PLR and DP optimizations concurrently...")
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("Both optimizations completed.")
