#include "viperfish.h"
#include <cfloat>
#include <random>
#include <algorithm>



enum LayerType {
    INPUT=1,
    OUTPUT=2,
    HIDDEN=3,
};


enum LossFunctionType {
    CROSS_ENTROPY_LOSS=1,
    MEAN_SQUARED_ERROR=2,
};

typedef void (*activation)(Matrix & m);
typedef void (*activation_grad)(Matrix & m);

Cublas _cublas;
Cublas *cublas=&_cublas;

inline Matrix matrix_new(size_t rows, size_t cols, std::vector<float> & data) {
    Matrix m(rows,cols,data);
    return m;
}
inline Matrix matrix_create(size_t rows, size_t cols) {
    Matrix  m(rows,cols);
    m.zero();
    return m;
}
inline Matrix createMatrixZeros(size_t rows, size_t cols) {
    return matrix_create(rows,cols);
}
inline void linear_act(Matrix& input) {
    
}
inline void linear_grad_act(Matrix& input) {
    input.fill(1.0f);
}
inline void softmax_act(Matrix& input) {
    input.softmax();    
}
inline void tanh_act(Matrix& input) {
    input.tanh();    
}
inline void tanh_grad_act(Matrix& input) {
    input.tanh_deriv();    
}
inline void sigmoid_act(Matrix& input) {            
    input.sigmoid();    
}
inline void sigmoid_grad_act(Matrix& input) {
    input.sigmoid_deriv();
}
inline void relu_act(Matrix& input) {    
    input.relu();    
}
inline void relu_grad_act(Matrix& input) {    
    input.relu_deriv();    
}


struct RandomNumbers
{
    typedef std::chrono::high_resolution_clock myclock;
    unsigned seed;
    std::default_random_engine generator;

    RandomNumbers() {
        myclock::time_point beginning = myclock::now();
        myclock::duration d = myclock::now() - beginning;
        seed = d.count();    
        generator = std::default_random_engine(seed);
    }

    void set_seed(unsigned s) { seed = s; }
    void reseed() {
        myclock::time_point beginning = myclock::now();
        myclock::duration d = myclock::now() - beginning;
        seed = d.count();    
        generator = std::default_random_engine(seed);
    }
    float random(float min=0.0f, float max=1.0f) {
        std::uniform_real_distribution<double> distribution(min,max);
        return distribution(generator);
    }
};


// random already exists somewhere.
float randr(float min = 0.0f, float max = 1.0f) {
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = myclock::now() - beginning;
    unsigned seed = d.count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(min,max);                
    return distribution(generator);
}


struct BoxMuller {
    float z0,z1;
    bool  do_generate;
    
    BoxMuller() {
        z0=z1=0.0;
        do_generate = false;
    }

    float generate() {
        float epsilon = FLT_MIN;
        float two_pi  = 2 * M_PI;
        do_generate = !do_generate;
        if(!do_generate) return z1;
        float u1 = randr();
        float u2 = randr();
        while(u1 <= epsilon) {
            u1 = randr();
            u2 = randr();
        }
        z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(two_pi * u2);
        z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(two_pi * u2);
        return z0;
    }
};

enum ActivationType {
    LINEAR=0,
    SIGMOID=1,
    RELU=2,
    TANH=3,
    SOFTMAX=4,
};

struct Connection;

struct Layer {
    LayerType        type;
    size_t           size;
    ActivationType   atype;
    Matrix        input;
    activation       activate_f;
    activation_grad  activate_grad_f;

    Layer(LayerType t, size_t s, ActivationType a) {
        type = t;
        size = s;
        atype = a;
        switch(a) {
            case LINEAR: activate_f = linear_act;
                         activate_grad_f = linear_grad_act;
                         break;
            case SIGMOID: activate_f = sigmoid_act;
                          activate_grad_f = sigmoid_grad_act;
                          break;                          
            case RELU:  activate_f = relu_act;
                        activate_grad_f = relu_grad_act;
                        break;
            case TANH:  activate_f = tanh_act;
                        activate_grad_f = tanh_grad_act;
                        break;
            case SOFTMAX:
                        activate_f = softmax_act;
                        activate_grad_f = linear_grad_act;
                        break;
        }
        input = Matrix(1,size);
        input.zero();
    }
    ~Layer() {

    }

    void Activate(Matrix& tmp) {     
        input = tmp;
        activate_f(input);                        
    }
    void Grad(Matrix & tmp) {
        activate_grad_f(tmp);        
    }

};


struct Connection {
    Layer * from,
          * to;

    Matrix weights;
    Matrix bias;

    Connection(Layer * from, Layer * to) {
        this->from = from;
        this->to   = to;
        weights = matrix_create(from->size,to->size);
        bias    = matrix_create(1,to->size);
        bias.fill(1.0f);

        BoxMuller bm;
        for(size_t i = 0; i < weights.rows(); i++)
            for(size_t j = 0; j < weights.cols(); j++)
                weights.set(i,j,bm.generate()/std::sqrt(weights.rows()));
        
        weights.upload_device();
    }
    ~Connection() {

    }
    void print() {
        weights.print();
        bias.print();
    }
};


struct ParameterSet {
    Matrix data;
    Matrix classes;
    LossFunctionType loss_function;
    size_t batch_size;
    float learning_rate;
    float search_time;
    float regularization_strength;
    float momentum_factor;
    size_t max_iters;
    bool shuffle;
    bool verbose;
    bool turbo;

    ParameterSet( Matrix &d, Matrix &c, 
                 size_t epochs, size_t bs,       
                 LossFunctionType loss=MEAN_SQUARED_ERROR,
                 float lr = 0.01, float st = 0.0,
                 float rs=0.0,float m=0.2, bool s=true, bool v=true, bool turbo_mode=true) {
            max_iters = epochs;            
            data = d;
            classes = c;
            loss_function = loss;
            batch_size = bs;
            learning_rate = lr;
            search_time = st;
            regularization_strength = rs;
            momentum_factor = m;
            shuffle = s;
            verbose = v;
            turbo = turbo_mode;
    }
};

struct Batch {
    Matrix example;
    Matrix training;

    Batch(Matrix & e, Matrix & c) {
        example    = e.eval();
        training   = c.eval();        
    }    
    Batch(const Batch & b) {
        example.resize(b.example.M,b.example.N);
        example.copy(b.example);
        training.resize(b.training.M,b.training.N);
        training.copy(b.training);
    }
    Batch& operator = (const Batch & b) {
        example.resize(b.example.M,b.example.N);
        example.copy(b.example);
        training.resize(b.training.M,b.training.N);
        training.copy(b.training);
        return *this;
    }
    void print() {
        std::cout << "--------------------------\n";
        example.print();
        training.print();
    }
};

// need to get gpu reduction
float sumsq(Matrix &tmp) {
    float total = 0;
    tmp.download_host();
    for(size_t i = 0; i < tmp.size(); i++)
        total += tmp[i]*tmp[i];
    return total;
}

struct Network {
    size_t num_features;
    size_t num_outputs;
    std::vector<Layer*> layers;
    std::vector<Connection*> connections;
    std::vector<std::vector<Batch>> batch_list;

    std::vector<Matrix> errori;
    std::vector<Matrix> dWi;
    std::vector<Matrix> dbi;
    std::vector<Matrix> regi;
    std::vector<Matrix> wTi;
    std::vector<Matrix> errorLastTi;
    std::vector<Matrix> fprimei;
    std::vector<Matrix> inputTi;
    std::vector<Matrix> dWi_avg;
    std::vector<Matrix> dbi_avg;
    std::vector<Matrix> dWi_last;
    std::vector<Matrix> dbi_last;

    Network(size_t num_features,
            std::vector<int64_t> & hidden,
            std::vector<ActivationType> & activations,
            size_t num_outputs,
            ActivationType output_activation
            )
    {
        assert(num_features > 0);
        assert(num_outputs > 0);
        this->num_features = num_features;
        this->num_outputs  = num_outputs;
        size_t num_hidden = hidden.size();
        size_t num_layers = 2 + num_hidden;
        layers.resize(num_layers);

        for(size_t i = 0; i < num_layers; i++)
        {
            Layer * ln = NULL;
            if(i == 0)
                ln = new Layer(INPUT,num_features, LINEAR);
            else if(i == num_layers-1)
                ln = new Layer(OUTPUT,num_outputs,output_activation);
            else
                ln = new Layer(HIDDEN, hidden[i-1], activations[i-1]);
            assert(ln != NULL);
            layers[i] = ln;
        }
        size_t num_connections = num_layers-1;
        for(size_t i = 0; i < num_connections; i++)
        {
            assert(layers[i] != NULL);
            assert(layers[i+1]!= NULL);
            Connection * c = new Connection(layers[i],layers[i+1]);
            connections.push_back(c);
        }
    }
    ~Network() {
        for(size_t i = 0; i < layers.size(); i++)
            delete layers[i];
        for(size_t i = 0; i < connections.size(); i++)
            delete connections[i];
    }

    size_t NumLayers() const { return layers.size(); }
    size_t NumConnections() const { return connections.size(); }
    size_t NumInputs() const { return num_features; }
    size_t NumOutputs() const { return num_outputs; }
    size_t LastLayer() const { return layers.size()-1; }

    void ForwardPass(Matrix& input) {
        assert(input.cols() == layers[0]->input.cols());
        layers[0]->input = input.eval();
        Matrix tmp;                
        for(size_t i = 0; i < connections.size(); i++)
        {                      
            tmp  = layers[i]->input*connections[i]->weights;
            tmp.addToEachRow(connections[i]->bias);            
            connections[i]->to->Activate(tmp);
        }        
    }
    float CrossEntropyLoss(Matrix& prediction, Matrix& actual, float rs) {
        float total_err = 0;
        float reg_err = 0;        
        total_err = (actual * log2(prediction)).sum();
        for(size_t i = 0; i < connections.size(); i++)
        {
            Matrix & weights = connections[i]->weights;
            reg_err += (hadamard(weights,weights)).sum();
        }
        return (-1.0f / actual.rows()*total_err) + rs*0.5f*reg_err;
    }
    float MeanSquaredError(Matrix& prediction, Matrix & actual, float rs) {
        float total_err = 0;
        float reg_err = 0;
        Matrix tmp = actual - prediction;
        total_err = hadamard(tmp,tmp).sum();
        for(size_t i = 0; i < connections.size(); i++)
        {
            Matrix & w = connections[i]->weights;
            reg_err += (hadamard(w,w)).sum();
        }
        return ((0.5f / actual.rows()) * total_err) + (rs*0.5f*reg_err);
    }
    Matrix& GetInput() {
        return layers[0]->input;
    }
    Matrix& GetOutput() {
        return layers[LastLayer()]->input;
    }
    // legacy
    std::vector<int> predict() {
        Layer* output_layer = layers[layers.size()-1];
        std::vector<int> prediction;
        prediction.resize(output_layer->input.rows());
        Matrix & input = output_layer->input;
        for(size_t i = 0; i < input.rows(); i++) {
            int max = 0;
            for(size_t j = 0; j < input.cols(); j++) {
                if(input.get(i,j) > input.get(i,max)) max = j;
            }
            prediction[i] = max;
        }
        return prediction;
    }
    float accuracy(Matrix & data, Matrix & classes) {
        ForwardPass(data);
        std::vector<int> p = predict();
        float num_correct = 0;
        for(size_t i = 0; i < data.rows(); i++) {
            if(classes.get(i,p[i]) == 1)
                num_correct++;
        }
        return 100*num_correct/classes.rows();
    }
    void shuffle_batches() {
        for(size_t i = 0; i < batch_list.size(); i++)
            std::random_shuffle(batch_list[i].begin(),batch_list[i].end());        
    }
    
    void generate_batches(size_t num_batches,
                            size_t batch_size,
                            Matrix & data,
                            Matrix & classes,
                            bool shuffle) {
        size_t rc = 0;
        batch_list.clear();        
        for(size_t i = 0; i < num_batches; i++) {
            std::vector<Batch> l;
            size_t cur_batch_size = batch_size;
            if(i == num_batches) {
                if( data.rows() % batch_size != 0) {
                    cur_batch_size = data.rows() % batch_size;
                }
            }
            for(size_t j = 0; j < cur_batch_size; j++) {
                Matrix e = data.get_row(rc);                
                Matrix c = classes.get_row(rc);                                                
                Batch b(e,c);                
                l.push_back(b);
                rc = rc + 1;
                rc = rc % data.rows();
            }
            batch_list.push_back(l);
        }        
        if(shuffle) shuffle_batches();
    }
    std::vector<Batch> generate_list( Matrix & data,
                        Matrix & classes,
                        bool shuffle) {
        size_t rc = 0;
        std::vector<Batch> batches;
        int num_batches = data.rows();
        
        for(size_t i = 0; i < num_batches; i++) {                        
            Matrix e = data.get_row(rc);                
            Matrix c = classes.get_row(rc);                                                
            Batch b(e,c);                
            batches.push_back(b);
            rc = rc + 1;
            rc = rc % data.rows();
         }         
         if(shuffle) {
             std::random_shuffle(batches.begin(),batches.end());
         }                        
         return batches;
    }
    void train_pattern(Batch & batch) {
        Matrix& example = batch.example;
        Matrix& target  = batch.training;                                    
        Matrix beforeOutputT;        
        ForwardPass(example);                                             
        for(size_t layer = layers.size()-1; layer > 0; layer--)
        {
            Layer* to = layers[layer];
            Connection* con = connections[layer-1];
            size_t hidden_layer = layer-1;                        
            if(layer == layers.size()-1) {                                                                                    
                errori[layer] = to->input - target;                                                                                                                                                                        
                beforeOutputT = con->from->input.t();                            
                dWi[hidden_layer] = beforeOutputT * errori[layer];                            
                dbi[hidden_layer] = errori[layer].eval();
            }                        
            else {                                                  
                wTi[hidden_layer] = connections[layer]->weights.t();
                errorLastTi[hidden_layer] = errori[layer+1] * wTi[hidden_layer];
                fprimei[hidden_layer] = con->to->input.eval();
                con->to->Grad(fprimei[hidden_layer]);                            
                errori[layer] = hadamard(errorLastTi[hidden_layer],fprimei[hidden_layer]);                                                                                    
                inputTi[hidden_layer] = con->from->input.t();                            
                dWi[hidden_layer] = inputTi[hidden_layer] * errori[layer];                            
                dbi[hidden_layer] = errori[layer].eval();
            }                     
        }                      
        for(size_t idx=0; idx < connections.size(); idx++) {                                                                                                                        
            dWi_avg[idx] = dWi[idx] + dWi_avg[idx];                                                
            dbi_avg[idx] = dbi[idx] + dbi_avg[idx];                                 
        }                     
    }
    void train_batch(size_t batch, size_t training) {        
        train_pattern(batch_list[batch][training]);
    }    
    
    void update(ParameterSet& ps, size_t epoch) {
        Matrix & data = ps.data;
        Matrix & classes = ps.classes;
        float currentLearningRate = ps.learning_rate;
        if(ps.search_time != 0) {
            currentLearningRate = ps.learning_rate / (1.0f + (epoch / ps.search_time));
        }                
        float clr = currentLearningRate / data.rows();        
        for(size_t idx = 0; idx < connections.size(); idx++)
        {                                                            
            dWi_avg[idx] = dWi_avg[idx] * clr;
            dbi_avg[idx] = dbi_avg[idx] * clr;                    
            regi[idx] = connections[idx]->weights * ps.regularization_strength;                                                                                                                        
            dWi_avg[idx] = regi[idx] + dWi_avg[idx];                                        
            dWi_last[idx] = dWi_last[idx] * ps.momentum_factor;                                                                                
            dbi_last[idx] = dbi_last[idx] * ps.momentum_factor;                                                            
            dWi_avg[idx] = (dWi_last[idx] + dWi_avg[idx]);
            dbi_avg[idx] = (dbi_last[idx] + dbi_avg[idx]);                                        
            dWi_avg[idx] = dWi_avg[idx] * -1.0f;
            dbi_avg[idx] = dbi_avg[idx] * -1.0f;                                                                                
            connections[idx]->weights = dWi_avg[idx] + connections[idx]->weights;
            connections[idx]->bias    = dbi_avg[idx] + connections[idx]->bias;                                        
            dWi_last[idx] = dWi_avg[idx] * -1.0f;
            dbi_last[idx] = dbi_avg[idx] * -1.0f;                                                                                
            dWi_avg[idx].zero();
            dbi_avg[idx].zero();                    
        }        
    }
    void report(size_t epoch, ParameterSet & ps) {
        Matrix & data = ps.data;
        Matrix & classes = ps.classes;

        if(ps.verbose == true) {
            if(epoch % 250 == 0 || epoch <= 1) {
                ForwardPass(data);
                if(ps.loss_function == CROSS_ENTROPY_LOSS) {
                    printf("EPOCH: %ld loss is %f\n",epoch, CrossEntropyLoss(GetOutput(),classes,ps.regularization_strength));
                }
                else {
                    printf("EPOCH: %ld loss is %f\n",epoch, MeanSquaredError(GetOutput(),classes,ps.regularization_strength));
                }
            }
        }
    }
    void clear_matrix() {
        errori.clear();
        dWi.clear();
        dbi.clear();
        regi.clear();

        for(size_t i = 0; i < connections.size(); i++) {
            errori.push_back(createMatrixZeros(1,layers[i]->size));
            dWi.push_back(createMatrixZeros(connections[i]->weights.rows(),
                                            connections[i]->weights.cols()));
            dbi.push_back(createMatrixZeros(1,connections[i]->bias.cols()));
            regi.push_back(createMatrixZeros(connections[i]->weights.rows(),
                                            connections[i]->weights.cols()));
        }
        errori.push_back(createMatrixZeros(1,layers[LastLayer()]->size));
        size_t num_hidden = layers.size()-2;        

        wTi.clear();
        errorLastTi.clear();
        fprimei.clear();
        inputTi.clear();
        for(size_t k = 0; k < num_hidden; k++)
        {
            wTi.push_back(createMatrixZeros(connections[k+1]->weights.cols(),connections[k+1]->weights.rows()));
            errorLastTi.push_back(createMatrixZeros(1,wTi[k].cols()));
            fprimei.push_back(createMatrixZeros(1,connections[k]->to->size));
            inputTi.push_back(createMatrixZeros(connections[k]->from->size,1));
        }
        dWi_avg.clear();
        dbi_avg.clear();
        dWi_last.clear();
        dbi_last.clear();

        for(size_t i = 0; i < connections.size(); i++) {
            dWi_avg.push_back(createMatrixZeros(connections[i]->weights.rows(),connections[i]->weights.cols()));            
            dbi_avg.push_back(createMatrixZeros(1,connections[i]->bias.cols()));
            dWi_last.push_back(createMatrixZeros(connections[i]->weights.rows(),connections[i]->weights.cols()));            
            dbi_last.push_back(createMatrixZeros(1,connections[i]->bias.cols()));
        }    
    }
    void batch(ParameterSet & ps) {
        
        Matrix & data = ps.data;
        Matrix & classes = ps.classes;

        clear_matrix();        
        size_t num_batches = data.rows() / ps.batch_size;

        if(data.rows() % ps.batch_size != 0) num_batches++;

        size_t epoch = 0;

        generate_batches(num_batches, ps.batch_size, data, classes, ps.shuffle);
        
        while(epoch <= ps.max_iters) {
            if(ps.shuffle) {
                shuffle_batches();
            }            
            for(size_t batch = 0; batch < num_batches; batch++) {            
                size_t cur_batch_size = ps.batch_size;                
                //epoch++;
                //if(epoch >= ps.max_iters) break;
                if(batch == num_batches) {
                    if(data.rows() % ps.batch_size != 0) {
                        cur_batch_size = data.rows() % ps.batch_size;
                    }
                }                                                
                for(size_t training = 0; training < cur_batch_size; training++)
                {
                    epoch++;
                    if(epoch >= ps.max_iters) break;
                    train_batch(batch,training);                    
                    if(epoch % 250) report(epoch,ps);                                                     
                }                                                        
            }                                   
            update(ps,epoch);
            
        }
    }
    void train(ParameterSet & ps) {
        
        Matrix & data = ps.data;
        Matrix & classes = ps.classes;
        clear_matrix();
        
        size_t num_batches = data.rows() / ps.batch_size;

        if(data.rows() % ps.batch_size != 0) num_batches++;

        size_t epoch = 0;

        //generate_batches(num_batches, ps.batch_size, data, classes, ps.shuffle);
        std::vector<Batch> batches = generate_list(data,classes,ps.shuffle);
        size_t batch = 0;
        while(epoch <= ps.max_iters) {            
            //epoch++;
            if(epoch >= ps.max_iters) break;
            if(ps.shuffle) std::random_shuffle(batches.begin(),batches.end());            
            for(batch=0; batch < batches.size(); batch++) 
            {
                epoch++;
                if(epoch >= ps.max_iters) break;
                train_pattern(batches[batch]);       
                if(epoch % 250 == 0 || epoch == 1) report(epoch,ps);                     
            }
            update(ps,epoch);                                                              
        }
    }
};

void XOR(ActivationType atype, float lt, float mf)
{

    std::vector<float> examples = {0,0,0,1,1,0,1,1};
    std::vector<float> training = {0,1,1,0};
    std::vector<float> examples_bp = {-1,-1,-1,1,1,-1,1,1};
    std::vector<float> training_bp = {-1,1,1,-1};

    Matrix e = matrix_new(4,2,examples);
    Matrix t = matrix_new(4,1,training);
    
    std::vector<int64_t> hidden = {16};
    std::vector<ActivationType> activations = {atype};
    Network net(2,hidden,activations,1,LINEAR);
    ParameterSet p(e,t,1000,4);
    p.learning_rate = lt;
    p.momentum_factor = mf;
    p.regularization_strength = 0;
    p.verbose = true;
    p.shuffle = true;
    //p.loss_function = CROSS_ENTROPY_LOSS;
    std::cout << "Cranium Online" << std::endl;
    net.train(p);

    std::cout << "Ready." << std::endl;    
    net.ForwardPass(e);
    Matrix &output = net.GetOutput();
    output.print();
}

void RunXORTest() {
    XOR(SIGMOID,0.3,0.9);
    XOR(TANH,0.1,0.9);
    XOR(RELU,0.1,0.9);      
}

int main(int argc, char * argv[]) {       
    RunXORTest();    
}
