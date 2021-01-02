const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
const { format } = require('path')
const { input } = require('@tensorflow/tfjs-node')

const img2x = imgPath => {
  const buffer = fs.readFileSync(imgPath)

  return tf.tidy(() => {
    const tensor = tf.node.decodeImage(new Uint8Array(buffer))
    const tensorResized = tf.image.resizeBilinear(tensor, [224, 224])
    return tensorResized.toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3])
  })
}

const getData = async (trainDir, outputDir) => {
  const classes = fs
    .readdirSync(trainDir)
    .filter(name => !name.includes('.'))

  fs.writeFileSync(`${outputDir}/classes.json`, JSON.stringify(classes))

  const data = []
  
  classes.forEach((dir, index) => {
    fs.readdirSync(`${trainDir}/${dir}`)
      .filter(name => name.match(/jpg$/))
      // .slice(0, 100)
      .forEach(name => {
        const imgPath = `${trainDir}/${dir}/${name}`
        data.push({
          imgPath,
          dirIndex: index
        })
      })
  })

  tf.util.shuffle(data)

  const ds = tf.data.generator(function* () {
    const count = data.length
    const batchSize = 32
    for(let start = 0;start < count;start += batchSize) {
      const end = Math.min(start + batchSize, count)
      console.log('当前批次', start)

      yield tf.tidy(() => {
        const inputs = []
        const labels = []

        for(let j = start;j < end;j++) {
          const { imgPath, dirIndex } = data[j]
          const tensor = img2x(imgPath)
          inputs.push(tensor)
          labels.push(dirIndex)
        }

        const imgTensor = tf.concat(inputs)
        const labelTensor = tf.tensor(labels)

        return {
          xs: imgTensor,
          ys: labelTensor
        }
      })
    }
  })

  return {
    ds,
    classes
  }
}

module.exports = getData