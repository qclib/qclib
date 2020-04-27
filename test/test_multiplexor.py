"""
Test quantum multiplexor
"""
from unittest import TestCase
import numpy as np
from qiskit import execute, Aer, QuantumRegister
from qclib import QuantumCircuit


class TestMultiplexor(TestCase):
    """
    Test rotation multiplexor
    """
    def test_ry_multiplexor(self):
        """
        Test ry multiplexor with 1 control
        """
        angles = [1.9823131728623846, 1.9106332362490186, 1.4274487578895312]
        state = [np.sqrt(0.1), np.sqrt(0.2), np.sqrt(0.4), np.sqrt(0.3)]

        quantum_register = QuantumRegister(2)
        circuit = QuantumCircuit(quantum_register)
        circuit.ry(angles[0], quantum_register[1])
        circuit.ry_multiplexor([angles[1], angles[2]], [quantum_register[0], quantum_register[1]])

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector(circuit)

        self.assertTrue(np.isclose(out_state, state).all())

    def test_ry_multiplexor_two_controls(self):
        """
        Test ry multiplexor with 2 controls
        """
        state = [-0.17320508075688773, 0.1414213562373095, 0.1414213562373095, 0.17320508075688773, 0.31622776601683794,
                 0.6324555320336759, -0.5477225575051661, -0.31622776601683794]
        angles = [2.4980915447965093, 1.5707963267948963, 1.4594553124539327, 4.9137469011750206, 1.7721542475852274,
                  2.214297435588181, 7.3303828583761845]

        quantum_register = QuantumRegister(3)
        circuit = QuantumCircuit(quantum_register)
        circuit.ry(angles[0], quantum_register[2])
        circuit.ry_multiplexor([angles[1], angles[2]], [quantum_register[1], quantum_register[2]])
        circuit.ry_multiplexor([angles[3], angles[4], angles[5], angles[6]], [quantum_register[0], quantum_register[1], quantum_register[2]])

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector(circuit)

        self.assertTrue(np.isclose(out_state, state).all())

    def test_ry_multiplexor_three_controls(self):
        """
        Test ry multiplexor with 3 controls
        """
        state = [0.12902927833615552, 0.08423196308016559, -0.18283995449379678, 0.02627261006398845,
                 -0.1331185900596331, 0.15481015750266888, 0.023124280344794825, -0.18820703794373955,
                 0.057985665584784025, -0.022116720064352086, 0.012549064836999088, -0.14109322062716054,
                 -0.19906369753939362, -0.14879961214578272, -0.18835650573719645, -0.09631697122169439,
                 0.04911446311674216, 0.023215623076891904, 0.003609160767876885, -0.21105799495704897,
                 0.21123734100731867, 0.12222494890959865, -0.051972096629911535, 0.07904930085233579,
                 0.1375516372093571, -0.08953997505482862, -0.2015825315013182, 0.16556454800321435, 0.144355742542705,
                 -0.07625732638712147, -0.16522883560358223, -0.052272030263435555, -0.12337416769163516,
                 -0.14387589386874872, -0.20458273007269542, -0.14343694333779683, -0.07188232470967756,
                 0.13370006016214994, 0.18970964276823934, -0.15706276794226895, -0.15820191915293164,
                 -0.019767154063920205, 0.055263108359105036, 0.05058397550933437, 0.12077590244829883,
                 -0.19743379564637195, 0.1643301090199749, -0.03487953261006956, -0.1693313157670747,
                 -0.016256287254526436, -0.15126069017936772, 0.0875544369594741, 0.08890067747047413,
                 -0.06502799120354168, -0.04339748998581084, 0.06806172850655538, -0.17685797890526841,
                 -0.046089563686412635, -0.06383604306025681, 0.0905375534929264, -0.08501480538821145,
                 -0.05359668336138149, 0.1272222952096178, 0.146288466732588]
        angles = [1.5034691734493482, 1.574065803584988, 1.30305358092108, 1.5517365900978408, 1.7043762300151155,
                  1.3337967652721452, 1.6597246971601167, 1.7172825508079275, 2.2566011029340736, 1.752806233800375,
                  1.315525032653771, 1.490245660950267, 2.0374415683442852, 1.0208363511049239, 1.5923366141900976,
                  1.7511073146157579, 1.4969286278133693, 2.3157245068693224, 1.4104014517897, 2.637813544705753,
                  0.7396157097061173, 2.01839007954852, 1.6304430861022865, 1.8436864301146576, 2.036894607175115,
                  0.8785698062036469, 1.255709901467735, 1.5978361892833608, 1.2648714362745828, 1.0898309236821233,
                  2.1851004868477313, 1.1567000720731837, 5.997755376906852, 4.561999904939534, -2.897085740450396,
                  -0.7287693046161536, -2.964176181408071, 7.566980448740325, 7.228565299004544, 0.8831140587867894,
                  -3.1073953323088848, 1.0490925788422885, 4.304802008391102, -1.154093062786493, 4.907967818794079,
                  -0.9719979500929016, 6.895983056703349, 8.00710900863286, 7.506139142921474, 4.128199154997695,
                  -1.3830597549089143, 6.531794646635902, 1.4824407815510017, -2.043594898451205, -0.41829780255261007,
                  6.4746043519503695, 5.233767485658046, -1.2630693598075264, 4.276837928532336, 6.793049240915537,
                  4.369844528816345, 7.408187785389272, 1.7099892276850819]

        quantum_register = QuantumRegister(6)
        circuit = QuantumCircuit(quantum_register)
        circuit.ry(angles[0], quantum_register[5])
        circuit.ry_multiplexor(angles[1:3], quantum_register[4:6])
        circuit.ry_multiplexor(angles[3:7], quantum_register[3:6])
        circuit.ry_multiplexor(angles[7:15], quantum_register[2:6])
        circuit.ry_multiplexor(angles[15:31], quantum_register[1:6])
        circuit.ry_multiplexor(angles[31:63], quantum_register[0:6])

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector(circuit)

        self.assertTrue(np.isclose(out_state, state).all())
