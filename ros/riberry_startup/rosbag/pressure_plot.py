import rosbag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import argparse
import numpy as np
import sys

# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold 0 --start 55.7 --duration 19.83
# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold -40 --start 100 --duration 20
# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold 0 --start 210 --duration 15
# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold -40 --start 250 --duration 30

def main():
    # --- 1. コマンドライン引数の設定 (すべてpositional) ---
    parser = argparse.ArgumentParser(description='rosbagの気圧データをmp4に変換します。')
    parser.add_argument('bagfile', help='読み込むrosbagファイルのパス')
    parser.add_argument('--threshold', type=float, default=-30.0, help='閾値の値を指定 (default: -30)')
    parser.add_argument('--start', default=0, type=float, help='開始時間（rosbag開始から何秒後か）')
    parser.add_argument('--duration', default=20, type=float, help='切り出す長さ（秒間）')
    parser.add_argument('--fps', type=int, default=15, help='出力動画のFPS (default: 15)')
    args = parser.parse_args()

    topic = '/yamaguchi_arm_2/fullbody_controller/pressure/39'
    output_file = 'pressure_viz___.mp4'

    # フォントの設定（Arialを指定）
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 24

    # --- 2. データの抽出 ---
    times, values = [], []
    print(f"Reading {args.bagfile}...")

    try:
        with rosbag.Bag(args.bagfile) as bag:
            # 全体の開始時刻を取得
            bag_start_time = bag.get_start_time()
            target_start = bag_start_time + args.start
            target_end = target_start + args.duration

            # read_messagesの引数で時間を指定せず、ループ内で判定する（互換性重視）
            for _, msg, t in bag.read_messages(topics=[topic]):
                curr_time = t.to_sec()

                # 指定範囲より前ならスキップ
                if curr_time < target_start:
                    continue
                # 指定範囲を超えたら終了
                if curr_time > target_end:
                    break

                rel_time = curr_time - target_start
                times.append(rel_time)
                values.append(msg.data)
    except Exception as e:
        print(f"Error reading bag: {e}")
        sys.exit(1)

    if not times:
        print("指定された範囲にデータが存在しませんでした。引数を確認してください。")
        return

    # --- 3. プロットの設定 ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.tick_params(axis='both', labelsize=18)
    line, = ax.plot([], [], lw=3, color='#007bff', zorder=4)

    ax.set_xlim(0, args.duration)
    ax.set_ylim(-50, 10)
    ax.set_xlabel('Time [s]', fontsize=32, fontweight='bold')
    ax.set_ylabel('Pressure [kPa]', fontsize=32, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, zorder=1)

    # 閾値の赤い線
    threshold_value = args.threshold
    ax.axhline(y=threshold_value, color='red', linestyle='-', lw=1.5, zorder=3)
    ax.text(args.duration * 0.01, threshold_value + 1, f'Threshold',
            color='red', fontsize=24, fontweight='bold',
            ha='left', zorder=3)

    plt.tight_layout(pad=0.3, h_pad=0, w_pad=0)

    # --- 4. アニメーションの設定 ---
    total_frames = int(args.duration * args.fps)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        current_time = frame / args.fps
        idx = np.searchsorted(times, current_time)
        line.set_data(times[:idx], values[:idx])
        return line,

    def print_progress(current_frame, total_frames):
        print(f"Exporting: {current_frame}/{total_frames} frames ({(current_frame/total_frames)*100:.1f}%)", end='\r')

    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)

    # --- 5. 保存 ---
    print(f"Exporting: {args.duration}s at {args.fps}fps...")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    ani.save(output_file, writer=writer, progress_callback=print_progress)
    plt.close()
    print(f"\nSuccessfully saved to {output_file}")

if __name__ == '__main__':
    main()
