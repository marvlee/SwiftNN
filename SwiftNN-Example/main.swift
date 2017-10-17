//
//  main.swift
//  SwiftNN-Example
//
//  Created by Marv on 10/17/17.
//  Copyright © 2017 Marvin Lee. All rights reserved.
//

import Foundation

func main() {

    // XOR
    let neuralNet = NeuralNet.init(sizes: [2,4,1])
    
    for i in 0..<10000 {
        neuralNet.backPropagation(inputs: [0.0, 0.0], expectedOutputs: [0.0], learningRate: 0.01)
        neuralNet.backPropagation(inputs: [0.0, 1.0], expectedOutputs: [1.0], learningRate: 0.01)
        neuralNet.backPropagation(inputs: [1.0, 0.0], expectedOutputs: [1.0], learningRate: 0.01)
        neuralNet.backPropagation(inputs: [1.0, 1.0], expectedOutputs: [0.0], learningRate: 0.01)
        
        print("epoch", i+1)
        print(neuralNet.feedForward(inputs: [0.0, 0.0]).map({ Float($0) }))
        print(neuralNet.feedForward(inputs: [0.0, 1.0]).map({ Float($0) }))
        print(neuralNet.feedForward(inputs: [1.0, 0.0]).map({ Float($0) }))
        print(neuralNet.feedForward(inputs: [1.0, 1.0]).map({ Float($0) }))
    }

}

main()
