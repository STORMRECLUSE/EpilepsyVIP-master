//
// Created by Victor Prieto on 21.03.16.
//
#ifndef EPILEPSYVIP2_SLIDING_STRUCTS_H
#define EPILEPSYVIP2_SLIDING_STRUCTS_H



#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace std;

//A dynamic class that supports changing length,
// and especially, circular shifts in index.
template <typename dtype>
class sliding_vector{
private:
    dtype* _data;
    // allocates and clears nums
    void _allocate(){
        if (_length>0) {
            _data = new dtype[_length];
            for(int i = 0; i < _length;i++){
                _data[i] =0;
            }
        }

    }


public:
    int _length;
    int _access_offset;

    sliding_vector(){
        _length = 0;
        _access_offset = 0;
        _allocate();
    }

    sliding_vector(int length){
        _length = max(length,0);
        _access_offset = 0;
        _allocate();
    }

    //What follows is a destructor for the sliding_vector class
    /*
    ~sliding_vector() {
        delete [] _data;

    }
    */
    void free_inside_data(){
        delete [] _data;
    }

    // change for one with destructor.
    // This one does NOT ensure memory is kept.
    sliding_vector<dtype> &operator = (const sliding_vector<dtype> & assigned_one){
        _data = assigned_one._data;
        _length = assigned_one._length;
        _access_offset = assigned_one._access_offset;
        return *this;
    }
    /*
    sliding_vector<dtype> &operator = (sliding_vector<dtype> assigned_one){
        _data = assigned_one._data;
        _length = assigned_one._length;
        _access_offset = assigned_one._access_offset;
    }

    sliding_vector(sliding_vector<dtype> &make_copy){
        _length =make_copy._length;
        _data = make_copy._data;
        _access_offset = make_copy._access_offset;
    }
    */

    dtype &operator[](int index) {
        //if((index < _length )&&(index>=(-_length))) {
            if(index + _access_offset >=0 )
                return _data[(index  + _access_offset)% _length];
            else
                return  _data[(index  + _access_offset)% _length + _length];
        //}
        //else {
        //    std::cout << "Index is too large or small!" << endl;
        //    exit(-1);
        //}
    }

    const dtype operator[](int index) const{
        //if((index < _length )&&(index>=(-_length))) {
            if(index + _access_offset >=0 )
                return _data[(index  + _access_offset)% _length];
            else
                return  _data[(index  + _access_offset)% _length + _length];
        //}
        //else {
        //    std::cout << "Index is too large or small!" << endl;
        //    exit(-1);
        //}
    }


};


//Similar to the sliding vector,
// the sliding matrix is a matrix class that
//

/*
template <typename dtype>
class sliding_matrix{

private:
    dtype* _data;
    void allocate(){
        _data = new dtype*[max(_n_cols,0)];
        _data[0] = new dtype[max(_n_cols*_n_rows,0)];
        if(( _n_rows > 0) && ( _n_cols >0)) {
/// Initialize the array of pointers to each column
            dtype* temp_pointer = _data[0];
            for (int n = 1; n < _n_cols; n++) {

                _data[n] = temp_pointer + _n_rows * n;
            }
        }
    }


public:
    int _n_rows;
    int _n_cols;
    int _access_offset_row;
    int _access_offset_col;

    sliding_matrix(){
        _length = 0;
        _access_offset_row = 0;
        _access_offset_col = 0;
        allocate();
    }

    sliding_matrix(int n_rows,int n_cols){
        _n_rows = n_rows;
        _n_cols = n_cols;
        _access_offset_row = 0;
        _access_offset_col = 0;
        allocate();
    }



    ~sliding_matrix() {
        if (_data != NULL){
            delete [] _data;
            if (_data[0] != NULL){
                delete  _data[0];
            }
        }


    }


    void free_memory(){

    }
    dtype &operator[](int row,int col) {
        if((row < _n_rows )&&(row>=(-_n_rows))&&(col < _n_cols )&&(row>=(-_n_cols))) {
            return _data[(row  + _access_offset_row)% _n_rows][(col  + _access_offset_col)% _n_cols];
        }
        else {
            std::cout << "Index is too large or small!" << endl;
            exit(-1);
        }
    }

    const dtype &operator()(int row,int col) const{
        if((row < _n_rows )&&(row>=(-_n_rows))&&(col < _n_cols )&&(row>=(-_n_cols))) {
            return _data[(row  + _access_offset_row)% _n_rows][(col  + _access_offset_col)% _n_cols];
        }
        else {
            std::cout << "Index is too large or small!" << endl;
            exit(-1);
        }
    }

};

*/
template <typename dtype>
class sliding_array{
private:
    sliding_vector<dtype>*_data;
    int _access_offset;

    void _allocate(){
        _data = new sliding_vector<dtype>[_num_vects];
        for(int i=0;i<_num_vects;i++){
            _data[i]= sliding_vector<dtype>(_length_vects);
        }

    }


public:
    int _num_vects;
    int _length_vects;

    sliding_array(){
        _num_vects = 0;
        _length_vects = 0;
        _access_offset = 0;
        _allocate();
    }

    sliding_array(int num_vects,int length_vects){
        _num_vects = max(num_vects,0);
        _length_vects = max(length_vects,0);
        _access_offset = 0;
        _allocate();
    }


    // uncomment for a destructor.
    // Note that the sliding_vector class also needs the destructor for this to fully work.
    /*
    ~sliding_array() {
        if (_data!= NULL) {
    //        for (int i = 0; i < _num_vects;i++){
    //            delete _data[i];
    //        }
    //        delete[] _data;
        }
    }
    */



    sliding_vector<dtype> &operator[](int row_ind) {
        if((row_ind < _num_vects )&&(row_ind>=(-_num_vects))) {
            return _data[row_ind];
        }
        else {
            std::cout << "row_ind is too large or small!" << endl;
            std::exit(-1);
        }
    }

    const sliding_vector<dtype> &operator[](int row_ind) const{
        if((row_ind < _num_vects )&&(row_ind>=(-_num_vects))) {
            return _data[row_ind];
        }
        else {
            std::cout << "row_ind is too large or small!" << endl;
            std::exit(-1);
        }
    }


    void free_inside_data(){
        if (_data!= NULL) {
            for (int i = 0; i < _num_vects;i++){
                _data[i].free_inside_data();
            }
            delete[] _data;
        }
    }

    void change_access_offset(int new_offset){
        _access_offset = new_offset;
        for(int i = 0; i < _num_vects;i++){
            _data[i]._access_offset = new_offset;
        }
    }

    void add_access_offset(int offset_offset){
        _access_offset += offset_offset;
        _access_offset %= _length_vects;

        for (int i = 0; i < _num_vects ; ++i) {
            _data[i]._access_offset = _access_offset;
        }
    }

};

template <class dtype>
ostream & operator << (ostream &os,
                       const sliding_vector<dtype> &print_vect) {

    os << scientific;

    os.precision(3);


    if((print_vect._length<=0) )
    {
        os<< "Invalid Dimensions of Matrix to print out!";
        return os ;
    }
    os << "[";
    for(int i =0; i<print_vect._length;i++){
        os << print_vect[i] << " ";
    }
    os << "]" <<endl;
    return os;
}

template <class dtype>
ostream & operator << (ostream &os,
                       const sliding_array<dtype> &print_arr) {

    os << scientific;

    os.precision(3);


    if((print_arr._num_vects<=0)||(print_arr._length_vects <=0) )
    {
        os<< "Invalid Dimensions of Matrix to print out!";
        return os ;
    }
    os << "["<<endl;
    for(int i =0; i<print_arr._num_vects;i++){
        os << print_arr[i] ;
    }
    os << "]"<<endl;
    return os;
}

#endif //EPILEPSYVIP2_SLIDING_STRUCTS_H
