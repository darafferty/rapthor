function objectToParsetString(obj, section) {
    var result = "[" + section + "]"
    for(var item in obj) {
        var value = obj[item]
        var vType = typeof value
        if(value === null) continue
        try{
            if (vType === 'object' && !Array.isArray(value) && value.hasOwnProperty('path')) {
                value = value.path
            } else if(vType === 'object' && Array.isArray(value) ) {
                value = "[" +  value.join(',') + "]"
            }
            else if(vType === 'object' && !value.hasOwnProperty('path'))  {
                throw "Unsupported type"
            } else {
                value = String(value)
            }

            result += "\n" + item + "=" + value
        } catch(e) {
            console.error(item, value, e)
            throw e
        }
    }
    return result
}
