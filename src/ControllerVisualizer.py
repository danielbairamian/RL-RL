import time
import clr
from rlbot.agents.base_agent import SimpleControllerState

clr.AddReference('ScpDriverInterface')
clr.AddReference('System')
clr.AddReference('System.Drawing')
clr.AddReference('System.Windows.Forms')

from ScpDriverInterface import ScpBus
from ScpDriverInterface import X360Controller


class Controller:

    def __init__(self, controller_number):
        self.CONTROLLER_NUM = controller_number
        self.bus = ScpBus()
        self.controller = X360Controller()
        print(type(self.bus))
        print(type(self.controller))
        time.sleep(.5)
        self.connect_controller(controller_number)

        self.output_report = bytearray(8)


    def connect_controller(self, controller_num):
        connected = self.bus.PlugIn(controller_num)
        if not connected:
            unplugged = self.bus.Unplug(controller_num)
            if not unplugged:
                exit("Could not unplug cont.")
            time.sleep(1)
            connected = self.bus.PlugIn(controller_num)
            if not connected:
                exit("Could not connect cont")

    time.sleep(1)

    '''
    steer: float = 0.0,
    throttle: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    jump: bool = False,
    boost: bool = False,
    handbrake: bool = False,
    use_item: bool = False
    
    Link to useful values: 
    https://github.com/irungentoo/Xiaomi_gamepad/blob/master/mi/ScpDriverInterface/X360Controller.cs
    
    '''
    def report(self, controller_state: SimpleControllerState):

        #print(controller_state.throttle)
        self.controller.RightTrigger = int(controller_state.throttle * 255)
        #self.controller.LeftTrigger = int(controller_state.throttle * 255)
        #print(controller_state.steer)
        self.controller.LeftStickX = 32767 * controller_state.steer
        self.controller.Buttons = 0
        if(controller_state.boost):
            self.controller.Buttons |= 1 << 12

        if(controller_state.jump):
            self.controller.Buttons |= 1 << 13
        result = self.bus.Report(self.CONTROLLER_NUM, self.controller.GetReport(), self.output_report)



if __name__ == "__main__":
    controller = Controller(1)
    controller.report()
    controller.report()
    time.sleep(10)
