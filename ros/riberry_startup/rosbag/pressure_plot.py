import rosbag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import argparse
import numpy as np
import sys

# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold -10 --start 55.7 --duration 19.83
# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold -35 --start 100 --duration 20
# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold -10 --start 210 --duration 15
# python pressure_plot.py rosbag_2026-04-18-21-04-13.bag --threshold -35 --start 255 --duration 19.83

def main():
    # --- 1. コマンドライン引数の設定 (すべてpositional) ---
    parser = argparse.ArgumentParser(description='rosbagの気圧データをmp4に変換します。')
    parser.add_argument('bagfile', help='読み込むrosbagファイルのパス')
    parser.add_argument('--threshold', type=float, default=-30.0, help='閾値の値を指定 (default: -30)')
    parser.add_argument('--start', default=0, type=float, help='開始時間（rosbag開始から何秒後か）')
    parser.add_argument('--duration', default=20, type=float, help='切り出す長さ（秒間）')
    parser.add_argument('--fps', type=int, default=15, help='出力動画のFPS (default: 15)')
    parser.add_argument('--save_img', action='store_true', help='終了時の静止画を保存し表示する')
    parser.add_argument('--save_svg', action='store_true', help='終了時の静止画をSVGとして保存する')
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

    # ロスバッグの記録時にタイムスタンプが一時的に固まって（バーストして）記録されたデータを、
    # 各ブロック間で均等に分散させて本来のサンプリング間隔に補正する
    clusters = []
    current_cluster = []
    for i, t in enumerate(times):
        if not current_cluster:
            current_cluster.append(i)
        else:
            prev_t = times[current_cluster[-1]]
            if t - prev_t > 0.01:
                clusters.append(current_cluster)
                current_cluster = [i]
            else:
                current_cluster.append(i)
    if current_cluster:
        clusters.append(current_cluster)

    spaced_times = np.zeros(len(times))
    for c_idx, cluster in enumerate(clusters):
        n_points = len(cluster)
        t_start = times[cluster[0]]
        if c_idx < len(clusters) - 1:
            t_next = times[clusters[c_idx + 1][0]]
        else:
            t_next = args.duration
        
        for step_i, idx in enumerate(cluster):
            spaced_times[idx] = t_start + step_i * (t_next - t_start) / n_points
    times = spaced_times.tolist()

    # --- 3. プロットの設定 ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.tick_params(axis='both', labelsize=18)
    line, = ax.plot([], [], lw=3, color='#555555', zorder=4)

    ax.set_xlim(0, args.duration)
    ax.set_ylim(-50, 10)
    label_style = {'fontsize': 32, 'fontweight': 'bold', 'color': '#222222'}
    ax.set_xlabel('Time [s]', **label_style)
    ax.set_ylabel('Pressure [kPa]', **label_style)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

    # 閾値の赤い線
    threshold_value = args.threshold
    ax.axhline(y=threshold_value, color='#999999', linestyle='--', lw=1.5, zorder=3)
    ax.text(args.duration * 0.01, threshold_value + 1, f'Threshold',
            color='#888888', fontsize=28, fontweight='normal',
            ha='left', zorder=3)

    # --- 状態表示用（Attached / Detached）の設定 ---
    # 配色の定義（見やすく、かつ主張しすぎない色）
    COLOR_ATTACHED = '#2E7D32'  # 深い緑
    COLOR_DETACHED = '#C62828'  # 深い赤
    COLOR_BG_INACTIVE = '#EEEEEE' # 非活性時の背景（薄いグレー）
    COLOR_TEXT_INACTIVE = '#AAAAAA' # 非活性時の文字（グレー）

    # 初期状態の判定
    initial_is_attached = values[0] < threshold_value
    # 状態ロック用の変数（関数内で nonlocal またはリストを使って保持）
    state_status = {'is_attached': initial_is_attached, 'locked': False}

    # ボックス付きテキストの配置
    bbox_style = dict(boxstyle='round,pad=0.3', lw=2)

    txt_att = ax.text(0.46, 1.15, 'Attached', transform=ax.transAxes,
                      fontsize=26, fontweight='bold', ha='right', va='center',
                      color='white' if initial_is_attached else COLOR_TEXT_INACTIVE,
                      bbox={**bbox_style, 
                            'facecolor': COLOR_ATTACHED if initial_is_attached else COLOR_BG_INACTIVE,
                            'edgecolor': COLOR_ATTACHED if initial_is_attached else COLOR_BG_INACTIVE})

    txt_det = ax.text(0.54, 1.15, 'Detached', transform=ax.transAxes,
                      fontsize=26, fontweight='bold', ha='left', va='center',
                      color='white' if not initial_is_attached else COLOR_TEXT_INACTIVE,
                      bbox={**bbox_style, 
                            'facecolor': COLOR_DETACHED if not initial_is_attached else COLOR_BG_INACTIVE,
                            'edgecolor': COLOR_DETACHED if not initial_is_attached else COLOR_BG_INACTIVE})

    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.99]) # 文字のために上部をさらに空ける

    # --- 4. アニメーションの設定 ---
    total_frames = int(args.duration * args.fps)

    def init():
        line.set_data([], [])
        # ロック状態をリセット
        state_status['is_attached'] = values[0] < threshold_value
        state_status['locked'] = False
        return line, txt_att, txt_det

    def update(frame):
        current_time = frame / args.fps
        idx = np.searchsorted(times, current_time)
        line.set_data(times[:idx], values[:idx])
        
        if idx > 0 and not state_status['locked']:
            current_val = values[idx-1]
            # 現在の状態と逆方向に閾値をまたいだかチェック
            changed = False
            if state_status['is_attached'] and current_val >= threshold_value:
                state_status['is_attached'] = False
                changed = True
            elif not state_status['is_attached'] and current_val < threshold_value:
                state_status['is_attached'] = True
                changed = True

            if changed:
                state_status['locked'] = True # 一度切り替わったら固定
                
                # スタイルの更新
                if state_status['is_attached']:
                    # Attachedを強調
                    txt_att.set_color('white')
                    txt_att.get_bbox_patch().set_facecolor(COLOR_ATTACHED)
                    txt_att.get_bbox_patch().set_edgecolor(COLOR_ATTACHED)
                    # Detachedを沈める
                    txt_det.set_color(COLOR_TEXT_INACTIVE)
                    txt_det.get_bbox_patch().set_facecolor(COLOR_BG_INACTIVE)
                    txt_det.get_bbox_patch().set_edgecolor(COLOR_BG_INACTIVE)
                else:
                    # Detachedを強調
                    txt_det.set_color('white')
                    txt_det.get_bbox_patch().set_facecolor(COLOR_DETACHED)
                    txt_det.get_bbox_patch().set_edgecolor(COLOR_DETACHED)
                    # Attachedを沈める
                    txt_att.set_color(COLOR_TEXT_INACTIVE)
                    txt_att.get_bbox_patch().set_facecolor(COLOR_BG_INACTIVE)
                    txt_att.get_bbox_patch().set_edgecolor(COLOR_BG_INACTIVE)
                
        return line, txt_att, txt_det

    def print_progress(current_frame, total_frames):
        print(f"Exporting: {current_frame}/{total_frames} frames ({(current_frame/total_frames)*100:.1f}%)", end='\r')

    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)

    # --- 5. 保存 ---
    if args.save_img or args.save_svg:
        # 1. 画像比率を4:3に変更 (例: 横8インチ, 縦6インチ)
        fig.set_size_inches(8, 6)
        
        # 2. Attached / Detached ラベルを非表示にする
        txt_att.set_visible(False)
        txt_det.set_visible(False)
        
        # ラベルが消えた分、上部の余白を再調整
        plt.tight_layout(pad=0.3)
        
        # 最終フレーム（期間の終わり）の状態をグラフに反映
        # total_frames - 1 で指定期間の最後のデータを描画します
        update(total_frames - 1)
        
        # 画像として保存 (ファイル名は bagファイル名などから生成)
        if args.save_svg:
            image_output = args.bagfile.replace('.bag', '_final.svg')
        else:
            image_output = args.bagfile.replace('.bag', '_final.pdf')
        plt.savefig(image_output)
        print(f"Successfully saved final frame to {image_output}")

    else:
        # 動画保存モード：save_imgが指定されていない場合のみ実行
        print(f"Exporting video: {args.duration}s at {args.fps}fps...")
        
        # FFMpegWriterの設定と保存処理[cite: 1]
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
        ani.save(output_file, writer=writer, progress_callback=print_progress)
        
        plt.close() # 動画保存時はウィンドウを閉じる
        print(f"\nSuccessfully saved video to {output_file}")

if __name__ == '__main__':
    main()
