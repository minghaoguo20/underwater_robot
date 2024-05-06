/*
  ******************************************************************************
  * @file           : JXZK_XLB_Raspi.C
  * @author  		: JXZK_124LAB	Lim-Tech Company
  * @brief          : Raspi interface program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 JXZK-124LAB	Lim-Tech Company.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided GPL.
  *
  ******************************************************************************
  */
#include  "JXZK_XLB_Raspi.h" 
#include  "JXZK_XLB_Parameter.h"  
//#include "stdint.h"
/*
 巡线任务当中AUV可能状态有巡线状态和识别状态
 Cam_Sign int->1Byte 规定0代表巡线(正常运动) 1代表识别(需要停止运行并且对应LED闪烁, LORA回传)
 巡线状态：
 机体坐标系右向为正
 Cam_Angle float->2Byte 16Bit 65536取整数 单位为° 有符号数 -65536/2 对应-180°， 65536/2对应180°线性
 Cam_Err   float->2Byte 待确定 可能是导线距离图片中心的大小， 单位为像素值 -240到240？
 识别状态：
 Cam_Flag int->1Byte 规定 1：纯绿色 2：纯蓝色 3：海洋动物-章鱼 4：海洋动物-鲨鱼 5：海洋动物-海龟
 数据包传输格式：
 开头(0xFE) + Cam_Sign + Cam_Flag + Cam_Angle(2) + Cam_Err(2) + 累加和校验(2) + 结束标志(0xFF)
*/
uint8_t Raspi_Package[10];
extern  XLB  AUH;
void Raspi_Caculate(void);
void Raspi_Img(uint8_t Raspi_chr)//树莓派图像处理函数
{
	static uint8_t index=0;
	if(index == 0){
		if(Raspi_chr == 0xFE){
			Raspi_Package[index] = Raspi_chr;
			index++;
		}
	}else if(index == 9){
		if(Raspi_chr == 0xFF){
			Raspi_Package[index] = Raspi_chr;
			index = 0;
			Raspi_Caculate();
		}else{
			index = 0;
		}
	}else{
		Raspi_Package[index] = Raspi_chr;
		index++;
	}
}
void Raspi_Caculate(void){
	uint8_t Cam_Sign, Cam_Flag;
	int16_t Cam_Angle, Cam_Err;
	// 累加和校验
	uint8_t i;
	uint16_t sum = Raspi_Package[7];
	sum = (sum<<8) + Raspi_Package[8];
	for(i=1;i<7;i++){
		sum = sum - Raspi_Package[i];
	}
	if(sum != 0) return;
	// 数据处理
	Cam_Sign = Raspi_Package[1];
	AUH.Cam_Sign = Cam_Sign;
	switch(Cam_Sign){
		//巡线任务
		case 0:
			Cam_Angle = Raspi_Package[3];
			Cam_Angle = (Cam_Angle<<8) + Raspi_Package[4];
			Cam_Err = Raspi_Package[3];
			Cam_Err = (Cam_Err<<8) + Raspi_Package[4];
			AUH.Cam_Angle = Cam_Angle * 180.0 / (65536/2);
			AUH.Cam_Err = Cam_Err * 240.0 / (65536/2);
			break;
		//识别任务
		case 1:
			Cam_Flag = Raspi_Package[2];
			AUH.Cam_Flag = Cam_Flag;
			break;
		//传输出错
		default: break;
	}
}
