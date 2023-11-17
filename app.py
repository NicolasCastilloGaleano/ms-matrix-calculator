from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# sumar matrices
@app.route('/sumar_matrices', methods=['POST'])
def sumar_matrices():
    try:
        data = request.get_json()
        matrices = data.get('matrices', [])
        print(matrices)

        # Verificar que haya al menos dos matrices para sumar
        if len(matrices) < 2:
            return jsonify({'error': 'Se requieren al menos dos matrices para la suma'}), 400

        # Verificar que las matrices tengan las mismas dimensiones
        shape_set = {np.array(mat).shape for mat in matrices}
        if len(shape_set) > 1:
            return jsonify({'error': 'Las matrices deben tener las mismas dimensiones para la suma'}), 400

        # Realizar la suma de las matrices
        resultado = np.sum(matrices, axis=0)

        return jsonify({'resultado': resultado.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# multiplicar matrices
@app.route('/multiplicar_matrices', methods=['POST'])
def multiplicar_matrices():
    try:
        data = request.get_json()
        matrices = data.get('matrices', [])

        # Verificar que haya al menos dos matrices para multiplicar
        if len(matrices) < 2:
            return jsonify({'error': 'Se requieren al menos dos matrices para la multiplicación'}), 400

        # Verificar que el número de columnas de la primera matriz sea igual al número de filas de la segunda matriz
        shape_set = {np.array(mat).shape for mat in matrices}
        if len(shape_set) > 1:
            return jsonify({'error': 'El número de columnas de la primera matriz debe ser igual al número de filas de la segunda matriz'}), 400

        # Realizar la multiplicación de las matrices
        resultado = np.dot(matrices[0], matrices[1])

        return jsonify({'resultado': resultado.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    
# calcular determinante
@app.route('/calcular_determinante', methods=['POST'])
def calcular_determinante():
    try:
        data = request.get_json()
        matriz = data.get('matriz', [])

        # Verificar que se proporciona al menos una matriz
        if not matriz:
            return jsonify({'error': 'Se requiere al menos una matriz para calcular el determinante'}), 400

        # Verificar que la matriz sea cuadrada (número de filas igual al número de columnas)
        shape = np.array(matriz).shape
        if len(shape) != 2 or shape[0] != shape[1]:
            return jsonify({'error': 'La matriz debe ser cuadrada para calcular el determinante'}), 400

        # Calcular el determinante de la matriz
        determinante = np.linalg.det(matriz)

        return jsonify({'determinante': determinante})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
#  valores y vectores propios
@app.route('/valores-vectores-propios', methods=['POST'])
def valoresVectoresPropios():
    try:
        data = request.get_json()
        matriz = data.get('matriz', [])

        # Verificar que se proporciona al menos una matriz
        if not matriz:
            return jsonify({'error': 'Se requiere al menos una matriz para calcular sus propiedades'}), 400

        # Verificar que la matriz sea cuadrada (número de filas igual al número de columnas)
        shape = np.array(matriz).shape
        if len(shape) != 2 or shape[0] != shape[1]:
            return jsonify({'error': 'La matriz debe ser cuadrada para calcular sus propiedades'}), 400

        # Calcular los valores y vectores propios de la matriz
        valores_propios, vectores_propios = np.linalg.eig(matriz)

        return jsonify({
            'valores_propios': valores_propios.tolist(),
            'vectores_propios': vectores_propios.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# diagonalizar matriz
@app.route('/diagonalizar_matriz', methods=['POST'])
def diagonalizar_matriz():
    try:
        data = request.get_json()
        matriz = data.get('matriz', [])

        # Verificar que se proporciona al menos una matriz
        if not matriz:
            return jsonify({'error': 'Se requiere al menos una matriz para diagonalizarla'}), 400

        # Verificar que la matriz sea cuadrada (número de filas igual al número de columnas)
        shape = np.array(matriz).shape
        if len(shape) != 2 or shape[0] != shape[1]:
            return jsonify({'error': 'La matriz debe ser cuadrada para diagonalizarla'}), 400

        # Calcular la diagonalización de la matriz
        valores_propios, matriz_P = np.linalg.eig(matriz)
        matriz_D = np.diag(valores_propios)
        matriz_P_inv = np.linalg.inv(matriz_P)

        return jsonify({
            'matriz_P': matriz_P.tolist(),
            'matriz_D': matriz_D.tolist(),
            'matriz_P_inversa': matriz_P_inv.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    
    
    
    
    
    


if __name__ == '__main__':
    app.run(debug=True)
