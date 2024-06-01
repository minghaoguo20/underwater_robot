from utils.gpio import *
import argparse
# # while True:
# com = serial.Serial(
#     port="/dev/ttyAMA0", baudrate=9600, stopbits=1, bytesize=8, parity="N"
# )
# msg = pack_lora_msg(2, 1, 44.4, 55.5)
# print(f"msg: {msg}")
# com.write(msg)

def test_communication(Cam_Sign = 0, Cam_Flag = 0, Cam_Angle = 0, Cam_Err = 0, receive = False):
    port = "/dev/ttyAMA0"  # "/dev/ttyS0"

    # 配置串口
    com = serial.Serial(port, 9600)  # 发送提示字符
    com.stopbits = 1
    com.bytesize = 8
    com.parity = "N"

    # Cam_Sign = 2
    # Cam_Flag = 19
    # Cam_Angle = -95.5
    # Cam_Err = -16.6

    msg = pack_lora_msg(
        Cam_Sign=Cam_Sign, Cam_Flag=Cam_Flag, Cam_Angle=Cam_Angle, Cam_Err=Cam_Err
    )
    # print(f"CAM_Sign: {Cam_Sign}, CAM_Flag: {Cam_Flag}, CAM_Angle: {Cam_Angle}, CAM_Err: {Cam_Err}")
    for m in msg:
        print(hex(m), end=" ")
    print()

    # com.write(b'underwater robot')
    # com.write(b'\xFE\x01\x04\x00\x00\x00\x00\x00\x05\xFF')
    com.write(msg)

    if receive:
        while True:
            count = com.inWaiting()
            if count != 0:
                recv = com.read(count)
                print(recv)  # 发回数据
                com.write(recv)
                com.flushInput()  # 清空接收缓冲区
                time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--sign', '-s', type=int, default=3, required=False, help="sign")
    parser.add_argument('--flag', '-f', type=int, default=0, required=False, help='flag')
    parser.add_argument('--angle', '-a', type=float, default=0, required=False, help="angle")
    parser.add_argument('--error', '-e', type=float, default=0, required=False, help="error")
    parser.add_argument('--receive', '-r', type=bool, default=False, required=False, help="receive")
    args = parser.parse_args()
    
    test_communication(args.sign, args.flag, args.angle, args.error, args.receive)
