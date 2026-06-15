'''
@Copyright (C): 2010-2019, Shenzhen Yahboom Tech
@Author: Malloy.Yuan
@Date: 2019-07-30 20:34:09
@LastEditors: Malloy.Yuan
@LastEditTime: 2019-08-08 16:10:46
'''

#  PID 제어 장치는 특정 시스템을 제어합니다.
#*****************************************************************#
#                      증분형 PID 시스템                              #
#*****************************************************************#
class IncrementalPID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
 
        self.PIDOutput = 0.0             #PID 컨트롤러 출력
        self.SystemOutput = 0.0          #시스템 출력 값
        self.LastSystemOutput = 0.0      #마지막 시스템 출력 값
 
        self.Error = 0.0                 #출력값과 입력값 사이의 편차
        self.LastError = 0.0             #마지막 편차
        self.LastLastError = 0.0           #마지막 마지막 편차
 
    #PID 컨트롤러 매개변수를 설정합니다.
    def SetStepSignal(self,StepSignal):
        self.Error = StepSignal - self.SystemOutput
        IncrementValue = self.Kp * (self.Error - self.LastError) +\
        self.Ki * self.Error +\
        self.Kd * (self.Error - 2 * self.LastError + self.LastLastError)

        self.PIDOutput += IncrementValue
        self.LastLastError = self.LastError
        self.LastError = self.Error

    #관성 시간 상수 InertiaTime을 사용하여 1차 관성계를 설정합니다.
    def SetInertiaTime(self,InertiaTime,SampleTime):
        self.SystemOutput = (InertiaTime * self.LastSystemOutput + \
            SampleTime * self.PIDOutput) / (SampleTime + InertiaTime)

        self.LastSystemOutput = self.SystemOutput
 
 
# *****************************************************************#
#                      위치 PID 시스템                              #
# *****************************************************************#
class PositionalPID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
 
        self.SystemOutput = 0.0
        self.ResultValueBack = 0.0
        self.PidOutput = 0.0
        self.PIDErrADD = 0.0
        self.ErrBack = 0.0
    
    #PID 컨트롤러 매개변수를 설정합니다.
    def SetStepSignal(self,StepSignal):
        Err = StepSignal - self.SystemOutput
        KpWork = self.Kp * Err
        KiWork = self.Ki * self.PIDErrADD
        KdWork = self.Kd * (Err - self.ErrBack)
        self.PidOutput = KpWork + KiWork + KdWork
        self.PIDErrADD += Err
        if self.PIDErrADD > 2000:
            self.PIDErrADD = 2000
        if self.PIDErrADD < -2500:
            self.PIDErrADD = -2500
        self.ErrBack = Err
        

    #관성 시간 상수 InertiaTime을 사용하여 1차 관성계를 설정합니다.
    def SetInertiaTime(self, InertiaTime,SampleTime):
           self.SystemOutput = (InertiaTime * self.ResultValueBack + \
           SampleTime * self.PidOutput) / (SampleTime + InertiaTime)

           self.ResultValueBack = self.SystemOutput
       