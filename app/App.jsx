import React, { PureComponent } from 'react'
import { Button, Progress } from 'antd'
import 'antd/dist/antd.css'
import * as tf from '@tensorflow/tfjs'
import { file2img, img2Tensor } from './utils'
import intro from './intro'

const DATA_URL = process.env.NODE_ENV === 'development' ?
  'http://localhost:9999' :
  './models'

class App extends PureComponent {
  constructor(props) {
    super(props)
    this.state = {}
  }

  async componentDidMount() {
    this.model = await tf.loadLayersModel(DATA_URL + '/model.json')
    this.CLASSES = await fetch(DATA_URL + '/classes.json').then(res => res.json())
  }

  async predict(file) {
    const img = await file2img(file)

    this.setState({
      imgSrc: img.src
    })
    
    setTimeout(() => {
      const result = tf.tidy(() => {
        const tensor = img2Tensor(img)
        return this.model.predict(tensor)
      })
  
      const finalResult = result.arraySync()[0]
        .map((score, index) => ({
          score,
          label: this.CLASSES[index]
        }))
        .sort((a, b) => b.score - a.score)
  
      this.setState({
        finalResult
      })
    }, 0)
  }

  renderResult(item) {
    const finalScore = Math.round(item.score * 100)
    return (
      <tr key={item.label}>
        <td style={{
          width: 80,
          padding: '5px 0'
        }}>{ item.label }</td>
        <td>
          <Progress percent={finalScore} status={finalScore === 100 ? 'success' : 'normal'}></Progress>
        </td>
      </tr>
    )
  }

  render() {
    const { imgSrc, finalResult } = this.state
    const finalItem = finalResult && { ...finalResult[0], ...intro[finalResult[0].label] }

    return (
      <div>
        <Button
          type="primary"
          style={{
            width: '100%'
          }}
          onClick={() => this.inputRef.click()}>选择图片识别</Button>
        <input
          ref={ref => this.inputRef = ref}
          type="file"
          style={{
            display: 'none'
          }}
          onChange={event => this.predict(event.target.files[0])}/>
        {
          imgSrc && (
            <div style={{
              marginTop: 20,
              textAlign: 'center'
            }}>
              <img
                src={imgSrc}
                style={{
                  maxWidth: '100%',
                  height: 300
                }} />
            </div>
          )
        }
        { finalItem && <div style={{
          marginTop: 20
        }}>识别结果：</div> }
        {
          finalItem && (
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                marginTop: 20
              }}>
              <img
                src={finalItem.icon}
                width="120"/>
              <div>
                <h2 style={{
                  color: finalItem.color
                }}>{ finalItem.label }</h2>
                <div style={{
                  color: finalItem.color
                }}>{ finalItem.intro }</div>
              </div>
            </div>
          )
        }
        {
          finalResult && (
            <div style={{
              marginTop: 20
            }}>
              <table style={{
                width: '100%'
              }}>
                <tbody>
                  <tr>
                    <td>类别</td>
                    <td>匹配度</td>
                  </tr>
                  {
                    finalResult.map(item => this.renderResult(item))
                  }
                </tbody>
              </table>
            </div>
          )
        }
      </div>
    )
  }
}

export default App