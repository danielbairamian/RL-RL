# automatically generated by the FlatBuffers compiler, do not modify

# namespace: flat

import flatbuffers

class DesiredGameState(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDesiredGameState(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DesiredGameState()
        x.Init(buf, n + offset)
        return x

    # DesiredGameState
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DesiredGameState
    def BallState(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .DesiredBallState import DesiredBallState
            obj = DesiredBallState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DesiredGameState
    def CarStates(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .DesiredCarState import DesiredCarState
            obj = DesiredCarState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DesiredGameState
    def CarStatesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DesiredGameState
    def BoostStates(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .DesiredBoostState import DesiredBoostState
            obj = DesiredBoostState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DesiredGameState
    def BoostStatesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DesiredGameState
    def GameInfoState(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .DesiredGameInfoState import DesiredGameInfoState
            obj = DesiredGameInfoState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DesiredGameState
    def ConsoleCommands(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .ConsoleCommand import ConsoleCommand
            obj = ConsoleCommand()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DesiredGameState
    def ConsoleCommandsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def DesiredGameStateStart(builder): builder.StartObject(5)
def DesiredGameStateAddBallState(builder, ballState): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(ballState), 0)
def DesiredGameStateAddCarStates(builder, carStates): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(carStates), 0)
def DesiredGameStateStartCarStatesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def DesiredGameStateAddBoostStates(builder, boostStates): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(boostStates), 0)
def DesiredGameStateStartBoostStatesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def DesiredGameStateAddGameInfoState(builder, gameInfoState): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(gameInfoState), 0)
def DesiredGameStateAddConsoleCommands(builder, consoleCommands): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(consoleCommands), 0)
def DesiredGameStateStartConsoleCommandsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def DesiredGameStateEnd(builder): return builder.EndObject()
