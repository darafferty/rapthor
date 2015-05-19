#
#   IDG	
#
IDG-XEON: XEON/IDG.cpp XEON/Proxies.cpp Common/RW.cpp Common/Init.cpp Common/Util.cpp
	g++ -o $@ $^ -ldl -fopenmp -ICommon

IDG-CUDA: CUDA/IDG.cpp CUDA/Proxies.cpp Common/RW.cpp Common/Init.cpp Common/Util.cpp libPowerSensor.o
	g++ -o $@ $^ -ldl -fopenmp -ICommon

#
#   Power sensor
#
libPowerSensor.o: PowerSensor/libPowerSensor.cc
	make -C PowerSensor libPowerSensor.o
	-@ln -sf PowerSensor/$@ $@

#
#	Util
#
EXEC=IDG-XEON IDG-CUDA

clean:
	@rm -f ${EXEC} *.so */*.so *.gch */*.ptx */*~ *~ libPowerSensor.o *.optrpt */*.optrpt
	-@make -s -C PowerSensor clean
