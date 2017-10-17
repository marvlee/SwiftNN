//
//  NeuralNet.swift
//  SwiftNN
//
//  Created by Marv on 8/25/17.
//  Copyright © 2017 Marvin Lee. All rights reserved.
//

import Foundation

public class NeuralNet {
    
    public class Neuron {
        
        var value : Double = 0.0
        var error : Double = 0.0
        
        var inputConnections : [Connection] = []
        var outputConnections : [Connection] = []
        
    }
    
    public class Connection {
        
        var input : Neuron!
        var output : Neuron!
        var weight : Double = 0.0
        
    }
    
    //MARK: Declaration
    
    var layers : [[Neuron]] = []
    var neurons : [Neuron] = []
    var connections : [Connection] = []
    
    var bias : Neuron!
    
    init(sizes: [Int]) { // layer sizes
        
        // create neurons
        for size in sizes {
            
            layers.append([])
            
            for i in 0..<size {
                
                let neuron = Neuron.init()
                
                neurons.append(neuron)
                layers[layers.count-1].append(neuron)
            }
        }
        
        // fully connect neurons
        for li in 0..<layers.count-1 { // layers
            
            for ni in 0..<layers[li].count { // neurons in layer
                
                for ni2 in 0..<layers[li+1].count { // neurons in next layer
                    
                    let connection = newConnection(input: layers[li][ni], output: layers[li+1][ni2])
                    addConnection(connection)
                }
            }
        }
        
        // bias neuron
        let biasNeuron = Neuron.init()
        biasNeuron.value = 1.0
        
        for neuron in neurons {
            
            let connection = newConnection(input: biasNeuron, output: neuron)
            addConnection(connection)
        }
        
        neurons.append(biasNeuron)
        
        bias = biasNeuron
    }
    
    //MARK: Connection
    
    // Create new connection
    func newConnection(input: Neuron, output: Neuron) -> Connection {
        
        let connection = Connection.init()
        connection.input = input
        connection.output = output
        connection.weight = randomWeight()
        
        return connection
    }
    
    func addConnection(_ connection: Connection) {
        
        connection.input.outputConnections.append(connection)
        connection.output.inputConnections.append(connection)
        
        connections.append(connection)
    }
    
    //MARK: Activation
    
    func feedForward(inputs: [Double]) -> [Double] {
        
        var outputs : [Double] = []
        
        bias.value = 1.0
        
        for li in 0..<layers.count {
            
            let layer = layers[li]
            
            for ni in 0..<layer.count {
                
                let neuron = layer[ni]
                
                if(li == 0) { // input
                    
                    neuron.value = inputs[ni]
                }
                else { // other than input
                    
                    var dotproduct = 0.0
                    
                    for connection in neuron.inputConnections {
                        
                        dotproduct += connection.input.value * connection.weight
                    }
                    
                    neuron.value = max(0, dotproduct) // rectified linear
                    
                    if(li == layers.count - 1) {
                        outputs.append(neuron.value)
                    }
                }
            }
        }
    
        return outputs;
    }
    
    func backPropagation(inputs: [Double], expectedOutputs: [Double], learningRate: Double) {
        
        feedForward(inputs: inputs)
        
        // change output error
        let outputLi = layers.count - 1
        
        for ni in 0..<layers[outputLi].count {
            
            let neuron = layers[outputLi][ni]
            let activation = neuron.value
            
            let error = (expectedOutputs[ni] - activation)
            neuron.error = error
        }
        
        // change hidden layer error
        for li in stride(from: layers.count - 2, through: 1, by: -1) {
            
            for ni in 0..<layers[li].count {
                
                let neuron = layers[li][ni]
                neuron.error = 0
                
                let gradient = (neuron.value > 0) ? 1.0 : 0.0
                
                for connection in neuron.outputConnections {
                    
                    neuron.error += connection.output.error * connection.weight * gradient
                }
            }
        }
        
        // update weights
        for connection in connections {
            
            let inputActivation = connection.input.value
            let outputError = connection.output.error
            
            connection.weight = connection.weight + (outputError * inputActivation * learningRate)
        }
    }
    
    //MARK: Numbers
    
    func randomWeight() -> Double {
        
        return randomNd(numOfInputs: 7)
    }
    
    func randomNd(numOfInputs: Int) -> Double { // normal distribution
        
        // random number from normal distribution
        
        let u = Double(arc4random() % 100000 + 1)/100000.0 //for precision
        let v = Double(arc4random() % 100000 + 1)/100000.0 //for precision
        
        //calculate the uniform distributed value with average of 0 and the standard deviation sigma of 1:

        let x = sqrt(-2 * log(u)) * cos(2 * .pi * v)
        
        //if needed add sigma and average for your target distribution like this:
        
        let sigmaValue = 1/sqrt(Double(numOfInputs))
        let mean = 0.0
        let y = x * sigmaValue + mean
        
        return y
    }
    
    func randomDouble() -> Double
    {
        return Double(arc4random()) / Double(UInt32.max)
    }
    
    func randomWeight1() -> Double {
        
        var value = randomDouble()
        
        if(randomDouble() < 0.5) {
            value = -value
        }
        
        return value
    }
    
}

